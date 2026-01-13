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
# M5: LieStrategyModule の指示文生成ロジック
# =========================================================
def get_strategy_decision_instructions(role: Role) -> str:
    """M5: M4の計画に基づき、嘘戦略の有無を決定するための指示を生成する"""
    instructions = (
        "【M5: 戦略的嘘の拡張決定】\n"
        "あなたは、M4モジュールで決定された主張の核（User Promptに記載）に基づき、その目標を達成するために「戦略的な嘘（欺瞞、情報隠蔽、ブラフを含む）」が必要か否かを評価・決定します。\n"
        "【戦略役的柔軟性の確保】：\n"
        "人狼陣営（人狼・狂人）である場合、パワープレイ（自身の役職のCOなど）といった高リスク戦略も、陣営の最終的な勝利に繋がる明確な論理的裏付けがある場合に限り採用を検討できます。単なる短期的な混乱を目的とするのではなく、多ターンにわたる戦略적利益を評価してください。\n"
    )

    # 1. 嘘の分類表 
    instructions += (
        "\n### 嘘の分類（Mechanism of Deception）\n"
        "あなたの決定する嘘は、以下のいずれかの分類に該当します。この分類から最も戦略的に有利なものを選択し、`lie_type`として出力してください。\n"
        "| 分類タイプ | 目的（欺瞞メカニズム） | 採用リスク/リターン |\n"
        "| :--- | :--- | :--- |\n"
        "| FactualDeception | 事実の偽造・虚偽: 役職、占い結果、襲撃ターゲットなど、ゲーム内の核となる事実について嘘をつく。 | 高リスク・高リターン |\n"
        "| ConsistencyBreak | 一貫性の偽装: 過去の行動や発言と矛盾しないように装う、または意図的に矛盾を突いて混乱を誘う。 | 中リスク・中リターン |\n"
        "| Omission/Distortion | 情報隠蔽・歪曲: 事実の一部を語らない、または意図的に解釈を誇張/矮小化して誘導する。 | 低リスク・低〜中リターン |\n"
        "| DirectBluff | 直接的なブラフ: 証拠なしで、次の行動（投票/襲撃）について誤った宣言をする。 | 中リスク・中リターン |\n"
    )
    
    # 2. 評価ロジックとリスク・リターン計算の具体化
    instructions += (
        "\n### 評価ロジックと目標の決定\n"
        "【意思決定の原則】：常に「リターン (R) - コスト (C) - リスク (D) = 期待価値 (EV) > 0」となる選択をしてください。\n"
        
        "1. 戦略的リスク・リターン評価: \n"
        "   - リターン (R): この戦略（嘘）が成功した場合、あなたの陣営の**勝利確率**はどれだけ向上するか？\n"
        "   - コスト (C): この嘘を実現するための発話の複雑性、他のエージェントを説得するための労力は？\n"
        "   - リスク (D)（嘘がバレる確率）: この嘘が露見した場合、あなたや勝利に必須の味方が吊られる確率は？（過去の自分の発言・行動との論理的矛盾も考慮）\n"
        
        "2. 論理的自己検証: 提案する嘘は、論理的一貫性を保ち、他者から反証されない完璧なブラフとして構築できますか？\n"

        "\n【嘘が不要なパターン (lie_used: false) の決定】\n"
        "以下のいずれかに該当する場合、`lie_used: false` を選択してください。**\n"
        "A. M4の主張の核（core_goal）が、嘘を使わなくても十分な効果を発揮し、EVが最大となる場合。\n"
        "B. 嘘を使うことのリスク (D) がリターン (R) を上回る場合。\n"
        "C. 提示できる有効な嘘戦略がなく、真実を語るのが最も安全な防御となる場合。\n"
    )
        
    
    # 3. 構造化出力の要求
    instructions += (
        "\n【出力形式】:\n"
        "思考プロセスは不要です。必ず以下のJSON形式で、唯一のオブジェクトを出力してください。\n"
        "`lie_used: false` の場合、`extended_goal`はM4の主張の核と同一の文字列を記述してください。\n"
        "```json\n"
        "{\n"
        "  \"lie_used\": true | false,\n"
        "  \"lie_type\": \"[上記分類表から選択したタイプを記述。lie_used: falseの場合は 'None']\",\n"
        "  \"risk_rating\": \"Low\" | \"Medium\" | \"High\" | \"None\",\n"
        "  \"extended_goal\": \"[嘘戦略を組み込んだ最終目標を記述。lie_used: falseの場合はM4の主張の核をそのまま記述]\"\n"
        "}\n"
        "```\n"
    )
    return instructions

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
    # M4の結果（主張の核）をテストデータから取得
    m4_output = data["modules"]["M4_UtterancePolicy"]["output"]
    core_goal = m4_output.get("core_goal", "なし")
    classification_type = m4_output.get("classification_type", "NONE")
    
    # M5の期待値を取得
    expected = data["modules"]["M5_LieStrategy"]["output"]
    
    my_id = context["my_id"]
    my_role_name = context["my_role"]
    agent_display_name = f"Agent[{my_id}]"
    my_role_enum = Role[my_role_name]
    m5_instructions = get_strategy_decision_instructions(my_role_enum)

    knowledge = "特になし"
    if my_role_enum == Role.SEER and "divined_result" in context:
        res = context["divined_result"]
        knowledge = f"占い結果: {res['target']} は {res['result']}"

    summary_str = "\n".join([f"- {s['agent']} -> {s.get('target','NONE')}: {s['summary']}" for s in context["recent_summaries"]])
    belief_summary = format_role_probabilities(context.get("role_probabilities", {}))
    
    # 指定されたフォーマットのプロンプト
    prompt_context = f"""
あなたは {my_role_name}の{agent_display_name} です。

{m5_instructions}

【あなたの役職情報】
知っていること情報: {knowledge}

【直近の会話要約】
{summary_str}

【主張の核】
{classification_type}:{core_goal}
"""

    #print(f"\n{'-'*20} [PROMPT: {data['scenario_id']}] {'-'*20}")
    #print(prompt_context.strip())
    #print(f"{'-'*60}")

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
        
        print(f"\n[LLM RAW RESPONSE]\n{content}")
        
        json_str = content[content.find('{'):content.rfind('}')+1]
        res_json = json.loads(json_str)
        
        actual_lie_used = res_json.get("lie_used")
        is_success = (actual_lie_used == expected["lie_used"])
        
        return {
            "id": data["scenario_id"],
            "success": is_success,
            "expected_lie": expected["lie_used"],
            "actual_lie": actual_lie_used,
            "extended_goal": res_json.get("extended_goal")
        }
    except Exception as e:
        return {"id": data["scenario_id"], "success": False, "error": str(e)}

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env or environment.")
        return

    current_dir = Path(__file__).parent
    scenario_dir = current_dir / "scenarios"
    scenario_files = list(scenario_dir.glob("*.json"))
    
    if not scenario_files:
        print(f"No scenario files found in '{scenario_dir}'.")
        return

    client = OpenAI(api_key=api_key)
    print(f"Evaluating M5 Lie Strategy for {len(scenario_files)} scenarios...\n")
    
    results = []
    for file_path in sorted(scenario_files):
        print(f"\nEvaluating: {file_path.name}")
        res = evaluate_scenario(client, file_path)
        results.append(res)
        
        if res.get("success"):
            print(f"\nRESULT: ✅ PASSED (lie_used: {res['actual_lie']})")
        else:
            print(f"\nRESULT: ❌ FAILED (Expected lie_used: {res.get('expected_lie')}, Actual: {res.get('actual_lie')})")
        print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    print(f"\nSummary: {passed}/{total} passed ({(passed/total)*100:.1f}%)")

if __name__ == "__main__":
    main()