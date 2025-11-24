# src/modules/lie_strategist.py (修正案)

from aiwolf_nlp_common.packet import Role
from utils.agent_logger import AgentLogger

class LieStrategyModule:
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        
    # メソッド名を変更: 戦略決定のための指示を生成する
    def get_strategy_decision_instructions(self, role: Role) -> str:
        """M2: 嘘戦略の有無と拡張された目標をJSONで決定させるための指示を生成する。"""
        if role == Role.WEREWOLF or role == Role.POSSESSED:
            instructions = (
                "【M2: 嘘戦略の拡張決定】\n"
                "あなたは、M3モジュールで決定された**主張の核**（User Promptに記載）に基づき、**嘘戦略**の要否を決定します。\n"
                
                "### 1. 嘘の使用要否と目標の拡張\n"
                "あなたの役職（人狼または狂人）の勝利のために、主張の核を達成するために**意図的な嘘や偽のCO**が必要か判断してください。\n"
                "嘘を使用する場合は `lie_used` を `true` とし、目標をより具体的で欺瞞的な内容に**拡張**してください。\n"
                
                "\n【出力形式】:\n"
                "思考プロセスは不要です。必ず以下の**JSON形式**で、**唯一のオブジェクト**を出力してください。\n"
                "```json\n"
                "{\n"
                "  \"lie_used\": true | false,\n"
                "  \"extended_goal\": \"[嘘戦略を組み込んだ最終的な目標を記述。嘘なしの場合は主張の核をそのまま記述]\"\n"
                "}\n"
                "```\n"
            )
            self.logger.debug("M2 Lie strategy decision instructions generated for external call.")
            return instructions
        return "" # 村人陣営の場合は指示を返さない