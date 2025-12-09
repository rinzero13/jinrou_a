from aiwolf_nlp_common.packet import Role
from utils.agent_logger import AgentLogger

class UtterancePolicyGenerator:
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def get_planning_prompt_instructions(self, role: Role) -> str:
        """M3: 主張の核の決定と応答方針をJSONで決定させるための指示を生成する。"""
        instructions = ""
        instructions += (
            "【M3: 初期計画と議論の焦点決定】\n"
            "あなたは、このターンにおける**議論の最優先目標（主張の核）**と、**直前の発話に対する応答の要否**を決定する戦略AIです。\n"
        )
        
        # --- 主張の分類表をプロンプトに組み込む ---
        instructions += (
            "### 発話の分類（Classification of Core Argument）\n"
            "あなたの**core_goal**は、以下の5つの分類のいずれかに該当するように決定してください。\n"
            "この分類から**最も戦略的に有利**なものを選択し、`classification_type`として出力してください。\n"
            "| 分類タイプ | 目的 | 具体例 |\n"
            "| :--- | :--- | :--- |\n"
            "| **Strategic: Accuse** | 村人陣営の疑いを集める/人狼陣営のターゲットを誘導する | 「プレイヤーXを人狼と告発する」「プレイヤーYの発言の矛盾を指摘する」 |\n"
            "| **Strategic: Claim/Deceive** | 自身を特定役職と偽る（人狼・狂人）/偽情報を流す | 「偽の占い師COを行う」「プレイヤーZを白だと偽証する」 |\n"
            "| **Defensive: Defend/Support** | 自身や味方を擁護し、吊りや襲撃を避ける | 「告発された味方を弁護する」「自分への疑いを明確に否定する」 |\n"
            "| **Informative: Reveal/Reason** | 既知の事実や推理過程を公開する（村人陣営） | 「真の占い結果を開示する」「現在の生存状況から合理的な推論を提示する」 |\n"
            "| **Meta: Observe/Skip** | 議論を観察する、情報が不足しているため沈黙する | 「議論を促す」「次のターンまで様子を見る」 |\n"
        )
        
        # 1. 主張の核の決定ロジック
        instructions += (
            "\n### 1. 主張の核（コア目標）の定義\n"
            "**指示:** あなたの役割の勝利目標とゲーム状況に基づき、**上記分類のいずれか**に該当する**最優先の戦略的目標**を簡潔に定義してください。\n"
            "**【議論の多様性・硬直化回避の要請】**：\n"
            "1. **直前の他プレイヤーの発話に対して、安易な同調（理由のない同意）を避けてください。**\n"
            "2. **直近3ターンの自分の発言や、議論が停滞している論点**の繰り返しを避け、**議論を次の段階に進める新しい視点や反証**を優先的に提示する目標を設定してください。\n"
            "**【論理的深度の要請】**：\n"
            "core_goalは、**このターンだけでなく、次の2～3ターンの議論展開を見据えた、深い論理的根拠**に基づき、一貫して追求できる目標を設定してください。\n"
        )

        # 2. 他プレイヤーへの応答の要否と方針の決定ロジック
        instructions += (
            "\n### 2. 他プレイヤーへの応答の要否と方針の決定\n"
            "**判断基準:** 直前の他プレイヤーの発話があなたへの直接的な質問や告発、またはあなたの主張の核に大きく影響するかを判断してください。\n"
            "**方針:** 応答が必須の場合は 'RESPOND_CRITICALLY' を、コア目標を優先する場合は 'PRIORITIZE_CORE' を選択してください。\n"
            "**【批判的応答の要請】**：\n"
            "'RESPOND_CRITICALLY' を選択した場合、応答目標は**単なる賛否の表明に留まらず、相手の発言の論理構造、根拠の妥当性、議論への影響度を深く分析し、反論または補強する**内容にしてください。\n"
        )
        
        # 3. 構造化出力の要求 (JSONに classification_type を追加)
        instructions += (
            "\n【出力形式】:\n"
            "思考プロセスは不要です。必ず以下の**JSON形式**で、**唯一のオブジェクト**を出力してください。\n"
            "```json\n"
            "{\n"
            "  \"classification_type\": \"[上記分類表から選択したタイプを記述]\",\n"
            "  \"core_goal\": \"[選択された分類に基づいた、最優先で達成すべき戦略的目標を簡潔に記述]\",\n"
            "  \"response_policy\": \"RESPOND_CRITICALLY\" | \"PRIORITIZE_CORE\",\n"
            "  \"response_target_id\": \"[応答対象のプレイヤーID。不要な場合は 'NONE']\"\n"
            "}\n"
            "```\n"
        )
        self.logger.logger.debug("M3 Planning instructions updated with argument classification.")
        return instructions