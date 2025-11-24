# src/modules/lie_strategist.py

from aiwolf_nlp_common.packet import Role
from utils.agent_logger import AgentLogger

class LieStrategyModule:
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        
    def get_lie_strategy_instructions(self, role: Role) -> str:
        """M2の戦略指示をLLMプロンプトに追加する。"""
        if role == Role.WEREWOLF or role == Role.POSSESSED:
            instructions = (
                "【M2: 嘘戦略の拡張】: 役職の勝利のために、発言の論理的整合性を保ちつつ、**意図的に誤解を招く発言や、戦略的な嘘**を交え、村人陣営の推論を妨害することを最優先してください。\n"
            )
            self.logger.debug("M2 Lie strategy instructions generated.")
            return instructions
        return ""