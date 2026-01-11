from typing import Dict, List, Any
from aiwolf_nlp_common.packet import Info, Role, Status, Setting
from utils.agent_logger import AgentLogger

class BeliefModel:
    """
    分析済みの発話意図、事実、および論理的整合性を考慮した信念モデル。
    """
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self.role_probabilities: Dict[str, Dict[str, float]] = {}
        self.my_agent_id: str = ""
        self.my_role: Role = None
        
        # 知識ベース：確定した役職COや占い結果を保存
        self.knowledge_base = {
            "co_map": {},      # {agent_id: Role}
            "divine_map": {},  # {target_id: result_species}
        }

        # 汎用的な発話意図ごとの尤度マトリックス
        # P(Intent | Role) の期待値
        self.INTENT_LIKELIHOODS = {
            "CO":      {"VILLAGER": 0.05, "SEER": 0.95, "WEREWOLF": 0.40, "POSSESSED": 0.60},
            "ATTACK":  {"VILLAGER": 0.50, "SEER": 0.40, "WEREWOLF": 0.70, "POSSESSED": 0.60},
            "DEFEND":  {"VILLAGER": 0.40, "SEER": 0.30, "WEREWOLF": 0.65, "POSSESSED": 0.60},
            "GUIDE":   {"VILLAGER": 0.55, "SEER": 0.70, "WEREWOLF": 0.50, "POSSESSED": 0.50},
            "DISRUPT": {"VILLAGER": 0.10, "SEER": 0.05, "WEREWOLF": 0.60, "POSSESSED": 0.80},
            "INQUIRY": {"VILLAGER": 0.60, "SEER": 0.50, "WEREWOLF": 0.40, "POSSESSED": 0.40},
            "NONE":    {"VILLAGER": 1.00, "SEER": 1.00, "WEREWOLF": 1.00, "POSSESSED": 1.00},
        }

    def initialize_probabilities(self, my_agent_id: str, my_role: Role, game_setting: Setting, all_agents: List[str]):
        self.my_agent_id = my_agent_id
        self.my_role = my_role
        num_others = len(all_agents) - 1
        role_num_map = getattr(game_setting, 'role_num_map', {})

        remaining_counts = {r: count for r, count in role_num_map.items()}
        remaining_counts[my_role] -= 1

        for agent_id in all_agents:
            if agent_id == my_agent_id: continue
            self.role_probabilities[agent_id] = {
                r.name: remaining_counts[r] / num_others for r in remaining_counts if remaining_counts.get(r, 0) > 0
            }

    def update_from_analyzed_data(self, game_info: Info, analyzed_talks: List[Dict[str, Any]]):
        """
        SpeechAnalyzerの要約データを元にベイズ更新を行う。
        """
        for talk in analyzed_talks:
            agent_id = talk['agent']
            if agent_id == self.my_agent_id: continue

            # 1. 知識ベースの更新
            if talk['intent'] == "CO" and talk['fact'] in [r.name for r in Role]:
                self.knowledge_base["co_map"][agent_id] = talk['fact']

            # 2. 論理性チェック (事実との整合性)
            integrity_score = self._validate_logic(talk, game_info)
            
            # 3. 尤度の取得と補正
            base_l = self.INTENT_LIKELIHOODS.get(talk['intent'], self.INTENT_LIKELIHOODS["NONE"])
            adjusted_l = self._adjust_likelihood_by_integrity(base_l, integrity_score)

            # 4. ベイズ更新の実行
            self._apply_bayesian_update(agent_id, adjusted_l)

            # 5. 関係性（ATTACK/DEFEND）による連動更新
            if talk['target'] != "NONE":
                self._apply_relational_update(agent_id, talk['target'], talk['intent'])

    def _validate_logic(self, talk: Dict[str, Any], game_info: Info) -> float:
        """発話の事実関係を検証し、0.0〜1.0でスコア化する。"""
        score = 0.5 # Default
        fact = talk['fact']
        target = talk['target']

        # 過去のCO情報との矛盾
        if talk['intent'] == "CO" and target in self.knowledge_base["co_map"]:
             if self.knowledge_base["co_map"][target] != fact:
                 score -= 0.3 # 前言撤回や矛盾
        
        # 占い結果等の客観的事実との矛盾（将来的に判定ロジックを強化可能）
        # 例: すでに襲撃された人を「占う」と言っている、など
        
        return max(0.0, min(1.0, score))

    def _adjust_likelihood_by_integrity(self, base_l: Dict[str, float], integrity: float) -> Dict[str, float]:
        """整合性スコアに基づき、村人陣営か人狼陣営かの尤度を増減させる。"""
        adjusted = base_l.copy()
        if integrity > 0.6: # 論理的
            for r in adjusted:
                if r in ["VILLAGER", "SEER", "BODYGUARD"]: adjusted[r] *= 1.2
        elif integrity < 0.4: # 矛盾
            for r in adjusted:
                if r in ["WEREWOLF", "POSSESSED"]: adjusted[r] *= 1.5
        return adjusted

    def _apply_bayesian_update(self, agent_id: str, likelihoods: Dict[str, float]):
        probs = self.role_probabilities.get(agent_id)
        if not probs: return

        total = 0.0
        for r_name in probs:
            l_val = likelihoods.get(r_name, 1.0)
            probs[r_name] *= l_val
            total += probs[r_name]
        
        if total > 0:
            for r_name in probs:
                probs[r_name] /= total

    def _apply_relational_update(self, agent_id: str, target_id: str, intent: str):
        """プレイヤー間の関係性（ライン）による確率の微調整"""
        if target_id not in self.role_probabilities: return
        
        p_agent_wolf = self.role_probabilities[agent_id].get("WEREWOLF", 0.2)
        
        if intent == "DEFEND":
            # 疑わしい人が庇っている相手は、連動して人狼確率を上げる（ラインの推定）
            self.role_probabilities[target_id]["WEREWOLF"] *= (1.0 + p_agent_wolf * 0.2)
        elif intent == "ATTACK":
            # 疑わしい人が攻撃している相手は、村人である可能性をわずかに上げる
            self.role_probabilities[target_id]["VILLAGER"] *= (1.0 + p_agent_wolf * 0.1)
        
        # 再正規化
        t_sum = sum(self.role_probabilities[target_id].values())
        if t_sum > 0:
            for r in self.role_probabilities[target_id]:
                self.role_probabilities[target_id][r] /= t_sum

    def get_top_beliefs_summary(self) -> str:
        summary_lines = ["--- 信念モデルによる役職推定 (論理性・関係性考慮) ---"]
        for agent_id, role_probs in self.role_probabilities.items():
            sorted_roles = sorted(role_probs.items(), key=lambda item: item[1], reverse=True)
            formatted = ", ".join([f"{role}: {prob:.1%}" for role, prob in sorted_roles if prob > 0.05])
            summary_lines.append(f"- {agent_id}: {formatted}")
        return "\n".join(summary_lines)