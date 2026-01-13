import json
import os
import glob
from pathlib import Path
from openai import OpenAI
from aiwolf_nlp_common.packet import Role

# .env ファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    # プロジェクトルート（testフォルダの親）にある.envを探す
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# =========================================================
# M4: UtterancePolicyGenerator の指示文生成ロジック
# =========================================================
def get_planning_prompt_instructions(role: Role) -> str:
    """M4: 主張の核の決定と応答方針を決定させるための指示を生成する"""
    instructions = (
        "【M4: 初期計画と議論の焦点決定】\n"
        "あなたは、このターンにおける議論の最優先目標（主張の核）と、直前の発話に対する応答の要否を決定する戦略AIです。\n"
    )
    
    instructions += (
        "### 発話の目的（Classification of Utterance Goal）\n"
        "あなたのcore_goalは、以下の6つの分類のいずれかに該当するように決定してください。\n"
        "この分類から最も戦略的に有利なものを選択し、`classification_type`として出力してください。\n"
        "| 分類タイプ | 目的（Goal） | 具体例（達成したい状態） |\n"
        "| :--- | :--- | :--- |\n"
        "| CO | 自己の正当性の確立/情報公開 | 自身の役職COや、真/偽の証拠提示により信頼を得る。 |\n"
        "| ATTACK | 他者の不審点/矛盾の指摘 | 特定プレイヤーへの告発を行い、議論の焦点を集める。 |\n"
        "| GUIDE | 処刑ターゲットの誘導/決定 | 特定の投票推奨を行い、村の合意形成を加速させる。 |\n"
        "| INQUIRY | 特定の情報収集/場への問いかけ | 未確定情報（他者の役職や信念）を知るための質問を投げる。 |\n"
        "| DEFEND | 味方の擁護/陣営の支援 | 告発された味方を弁護し、議論を自身に有利な方向へ戻す。 |\n"
        "| DISRUPT | 情報撹乱・場全体の混乱 | 議論の焦点を意図的にずらし、村の判断力を低下させる。 |\n"
    )
    
    instructions += (
        "\n### 1. 主張の核（コア目標）の定義\n"
        "指示: あなたの役割の勝利目標とゲーム状況に基づき、上記分類のいずれかに該当する最優先の戦略的目標を簡潔に定義してください。\n"
        "【議論の多様性・硬直化回避の要請】：\n"
        "1. 直前の他プレイヤーの発話に対して、安易な同調（理由のない同意）を避けてください。\n"
        "2. 直近3ターンンの自分の発言や、議論が停滞している論点の繰り返しを避け、議論を次の段階に進める新しい視点や反証を優先的に提示する目標を設定してください。\n"
        "【論理的深度の要請】：\n"
        "core_goalは、このターンだけでなく、次の2～3ターンの議論展開を見据えた、深い論理的根拠に基づき、一貫して追求できる目標を設定してください。\n"
        "【役職ごとの発話方針】：\n"
        "占い師は、占い結果をもとに、真実を明らかにし、村の合意形成を支援する。\n"
        "人狼は、自分の役職を隠蔽し、自陣営の勝利に繋がるような議論の方向性を巧みに操作する。\n"
        "狂人は、役職騙りなどの情報操作を巧みに使い、村の混乱を誘導し、自陣営の勝利を図る。\n"
        "村人は、情報の整合性を確認し、真実を明らかにし、村の合意形成を支援する。\n"
    )

    instructions += (
        "\n### 2. 他プレイヤーへの応答の要否と方針の決定\n"
        "判断基準: 直前の他プレイヤーの発話があなたへの直接的な質問や告発、またはあなたの主張の核に大きく影響するかを判断してください。\n"
        "方針: 応答が必須の場合は 'RESPOND_CRITICALLY' を、コア目標を優先する場合は 'PRIORITIZE_CORE' を選択してください。\n"
        "【批判的応答の要請】：\n"
        "'RESPOND_CRITICALLY' を選択した場合、応答目標は単なる賛否の表明に留まらず、相手の発言の論理構造、根拠の妥当性、議論への影響度を深く分析し、反論または補強する内容にしてください。\n"
    )
    
    instructions += (
        "\n【出力形式】:\n"
        "思考プロセスは不要です。必ず以下のJSON形式で、唯一のオブジェクトを出力してください。\n"
        "```json\n"
        "{\n"
        "  \"classification_type\": \"[上記分類表から選択したタイプを記述]\",\n"
        "  \"core_goal\": \"[選択された分類に基づいた、最優先で達成すべき戦略的目標を簡潔に記述]\",\n"
        "  \"response_policy\": \"RESPOND_CRITICALLY\" | \"PRIORITIZE_CORE\",\n"
        "  \"response_target_id\": \"[応答対象のプレイヤーID。不要な場合は 'NONE']\"\n"
        "}\n"
        "```\n"
    )
    return instructions

# =========================================================
# 評価用ヘルパー関数
# =========================================================
def format_role_probabilities(probs_dict: dict) -> str:
    if not probs_dict: return "推定データなし"
    lines = []
    for agent_id, roles in probs_dict.items():
        sorted_roles = sorted(roles.items(), key=lambda x: x[1], reverse=True)
        p_str = ", ".join([f"{role}({prob*100:.1f}%)" for role, prob in sorted_roles])
        lines.append(f"- Agent[{agent_id}]: {p_str}")
    return "\n".join(lines)

def evaluate_scenario(client, scenario_path):
    with open(scenario_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    context = data["context"]
    expected = data["modules"]["M4_UtterancePolicy"]["output"]
    
    my_id = context["my_id"]
    my_role_name = context["my_role"]
    agent_display_name = f"Agent[{my_id}]"
    my_role_enum = Role[my_role_name]
    m4_instructions = get_planning_prompt_instructions(my_role_enum)

    knowledge = "特になし"
    if my_role_enum == Role.SEER and "divined_result" in context:
        res = context["divined_result"]
        knowledge = f"占い結果: {res['target']} は {res['result']}"

    summary_str = "\n".join([f"- {s['agent']} -> {s.get('target','NONE')}: {s['summary']}" for s in context["recent_summaries"]])
    belief_summary = format_role_probabilities(context.get("role_probabilities", {}))
    
    # 指定されたフォーマットのプロンプト
    prompt_context = f"""
あなたは {my_role_name}の{agent_display_name} です。

{m4_instructions}

【現在のゲーム情報】
日目: Day {context['day']}
生存者: {', '.join(context['alive_players'])}
前回追放: {context.get('executed', 'なし')}
前回襲撃: {context.get('attacked', 'なし')}
あなたの役職の知っていること: {knowledge}
--- 信念モデルによる推定 ---
{belief_summary}

【直近の会話要約】
{summary_str}
"""
    # --- 実際の送信プロンプトを表示 ---
#    print(f"\n{'-'*20} [PROMPT: {data['scenario_id']}] {'-'*20}")
#    print(prompt_context.strip())
#    print(f"{'-'*60}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは戦略的な計画をJSON形式で出力する専門のAIです。"},
                {"role": "user", "content": prompt_context}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        
        # LLMの生の出力を表示
        print(f"\n[LLM RAW RESPONSE]\n{content}")
        
        json_str = content[content.find('{'):content.rfind('}')+1]
        res_json = json.loads(json_str)
        
        actual_type = res_json.get("classification_type")
        is_success = (actual_type == expected["classification_type"])
        
        return {
            "id": data["scenario_id"],
            "success": is_success,
            "expected": expected["classification_type"],
            "actual": actual_type
        }
    except Exception as e:
        return {"id": data["scenario_id"], "success": False, "error": str(e)}


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment.")
        return

    # このプログラムがある場所を基準に scenarios フォルダを探す
    current_dir = Path(__file__).parent
    scenario_dir = current_dir / "scenarios"
    scenario_files = list(scenario_dir.glob("*.json"))
    
    if not scenario_files:
        print(f"No scenario files found in '{scenario_dir}'.")
        return

    client = OpenAI(api_key=api_key)
    print(f"Evaluating {len(scenario_files)} scenarios from {scenario_dir}...\n")
    
    results = []
    for file_path in sorted(scenario_files):
        # 進行状況を見やすくするために区切りを追加
        print(f"\nEvaluating: {file_path.name}")
        res = evaluate_scenario(client, file_path)
        results.append(res)
        
        # 判定結果の表示
        if res.get("success"):
            print(f"\nRESULT: ✅ PASSED (Type: {res['actual']})")
        else:
            print(f"\nRESULT: ❌ FAILED (Expected: {res.get('expected')}, Actual: {res.get('actual')})")
            if "error" in res:
                print(f"ERROR DETAILS: {res['error']}")
        print(f"{'='*60}")

    # 全体サマリーの表示
    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*40}")
    print(f"Evaluation Summary:")
    print(f"Total Scenarios: {total}")
    print(f"Passed:          {passed}")
    print(f"Failed:          {total - passed}")
    print(f"Success Rate:    {(passed/total)*100:.1f}%" if total > 0 else "N/A")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()