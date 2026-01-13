import json
import os
import glob
from pathlib import Path
from openai import OpenAI
from aiwolf_nlp_common.packet import Role

# .env ファイルから環境変数を読み込む
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# =========================================================
# M7: LogicalConsistencyChecker プロンプト生成ロジック
# (ご提示のプログラムに準拠)
# =========================================================
def create_checker_prompt(context, virtual_talk, my_id):
    """矛盾判定用のLLMプロンプトを生成する。"""
    
    # テストデータ(context)から情報を抽出
    day = context.get("day", 1)
    alive_agents_list = context.get("alive_players", [])
    executed_name = context.get("executed", "なし")
    attacked_name = context.get("attacked", "なし")
    raw_talks = context.get("raw_talks", [])
    
    # プレイヤー名の表示設定 (テスト用なのでIDをそのまま使用)
    speaker_name = f"Agent[{my_id}]"

    # 判定基準のシステムメッセージ (ご提示のロジックを完全反映)
    system_message = (
        "あなたは人狼ゲームのAIの発話を分析する専門家です。\n"
        "以下の【現在のゲーム情報】、【発話履歴】、【評価対象の発話】に基づき、"
        "その発話が**論理的一貫性を保っているか**を判定してください。\n"
        "特に以下の5点に基づき、矛盾の有無を厳格に評価すること。論理的な飛躍も矛盾と見なします。\n"
        "1. **客観的事実・状況との関係** (ルール、既知の事実、可能性の範囲との矛盾)\n"
        "2. **発話者の行動・役職との関係** (過去の投票・CO・真の役職との矛盾)\n"
        "3. **他のプレイヤーの発話との関係** (応答性、主張の根拠、矛盾の指摘の妥当性)\n"
        "4. **ゲーム外の矛盾** (発話ルールに従っていない、不自然な発話)\n"
        "5. **議論の深化と論理的継続性**:\n"
        "   - 発話内容が、**理由のない単純な同調**、または**直近の議論内容の単なる繰り返し**になっていないか？\n"
        "   - **2〜3ターン先の議論展開**を見据えた、**深い論理的根拠**を伴う主張か？理由を述べずに同意する場合、論理の一貫性なしと判定する。\n"
        
        "**【戦略の柔軟性に関する判定基準】**:\n"
        "発話が**自身の真の役職や勝利目標を間接的に示唆する内容**であっても、それが**ゲーム状況全体から見て、戦略的に一貫性のある論理的根拠**に基づき、"
        "**論理的に成立しうる高リスク戦略**であるならば、論理的一貫性ありと判定してください。ただし、その戦略的根拠が発話内容から論理的に推測できる必要があります。\n"

        "発言者が人狼陣営の場合でも、発言自体に論理的な矛盾があれば、論理的一貫性なしと判定してください。\n"
    )
    
    my_role_name = context["my_role"]
    my_role_enum = Role[my_role_name]
    summary_str = "\n".join([f"- {s['agent']} -> {s.get('target','NONE')}: {s['summary']}" for s in context["recent_summaries"]])
    knowledge = "特になし"
    if my_role_enum == Role.SEER and "divined_result" in context:
        res = context["divined_result"]
        knowledge = f"占い結果: {res['target']} は {res['result']}"



    # ユーザープロンプト (直近5件の履歴をJSON形式で含める)
    user_prompt = (
        f"【現在のゲーム情報】\n"
        f"日目: {day}, 生存者: {', '.join(alive_agents_list)}\n"
        f"確定情報: 前回追放: {executed_name} / 前回襲撃: {attacked_name}\n"
        f"評価対象が知っていること：{knowledge}\n"
        f"【直近の会話要約】\n"
        f"{summary_str}\n"
        f"【評価対象の発話】\n"
        f"発話者: {speaker_name}\n"
        f"発話内容: {virtual_talk}\n\n"
        f"**出力形式:**\n"
        f"必ず以下のJSON形式で結果を返してください。reasoningには、論理的矛盾がある場合の詳細な理由と、**修正するための具体的な指示**を記述してください。\n"
        f"{{ \"is_consistent\": true | false, \"reasoning\": \"評価の詳細な理由と修正指示を記述\" }}\n"
    )

    return system_message, user_prompt

# =========================================================
# 評価実行ヘルパー
# =========================================================
def evaluate_scenario(client, scenario_path):
    with open(scenario_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    context = data["context"]
    # M6（生成された発話）とM7期待値を取得
    virtual_talk = data["modules"]["M6_UtteranceGeneration"]["output"]
    expected = data["modules"]["M7_ConsistencyCheck"]["output"]
    
    system_message, user_prompt = create_checker_prompt(context, virtual_talk, context["my_id"])

    # プロンプトの表示
    #print(system_message)
    #print(user_prompt)
    
    print(f"\n{'-'*20} [PROMPT: {data['scenario_id']}] {'-'*20}")
    print(f"Speaker Role: {context['my_role']}")
    print(f"Target Talk: {virtual_talk}")
    print(f"{'-'*60}")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        
        print(f"\n[LLM RAW RESPONSE]\n{content}")
        
        json_str = content[content.find('{'):content.rfind('}')+1]
        res_json = json.loads(json_str)
        
        actual_consistent = res_json.get("is_consistent")
        is_success = (actual_consistent == expected["is_consistent"])
        
        return {
            "id": data["scenario_id"],
            "success": is_success,
            "expected": expected["is_consistent"],
            "actual": actual_consistent,
            "reasoning": res_json.get("reasoning")
        }
    except Exception as e:
        return {"id": data["scenario_id"], "success": False, "error": str(e)}

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env.")
        return

    client = OpenAI(api_key=api_key)
    current_dir = Path(__file__).parent
    scenario_dir = current_dir / "scenarios"
    scenario_files = list(scenario_dir.glob("*.json"))
    
    if not scenario_files:
        print(f"No scenario files found in '{scenario_dir}'.")
        return

    print(f"Evaluating M7 Consistency Checker with {len(scenario_files)} scenarios...\n")
    
    results = []
    for file_path in sorted(scenario_files):
        print(f"\nEvaluating: {file_path.name}")
        res = evaluate_scenario(client, file_path)
        results.append(res)
        
        if res.get("success"):
            print(f"\nRESULT: ✅ PASSED (Consistency: {res['actual']})")
        else:
            print(f"\nRESULT: ❌ FAILED (Expected: {res.get('expected')}, Actual: {res.get('actual')})")
        print(f"{'='*60}")

    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    print(f"\nFinal Summary: {passed}/{total} passed ({(passed/total)*100:.1f}%)")

if __name__ == "__main__":
    main()