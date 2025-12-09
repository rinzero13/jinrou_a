from typing import Dict, List, Any
from aiwolf_nlp_common.packet import Info, Role, Status, Setting, Talk, Judge
from utils.agent_logger import AgentLogger

POSSIBLE_ROLES = [
    Role.VILLAGER,
    Role.WEREWOLF,
    Role.SEER,
    Role.BODYGUARD,
    Role.MEDIUM,
    Role.POSSESSED,
]

class BeliefModel:
    """他プレイヤーの役職確率を推定し、行動決定の根拠を提供するモデル。"""
    # 尤度パラメータ
    P_TRUE = 0.95        # 真の役職者が正直に行動する基本確率
    P_FALSE_CO = 0.05    # 村人陣営が偽COをするノイズ確率
    P_WOLF_LIE = 0.85    # 人狼陣営が戦略的な嘘をつく確率 (偽CO, 黒出しなど)
    P_WOLF_NO_LIE = 0.15 # 人狼陣営が戦略的な嘘をつかない確率 (白出しなど)
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        # {AgentID: {Role.name: probability}} の形式で確率を保持
        # 例: {"kanolab2": {"VILLAGER": 0.8, "WEREWOLF": 0.1, ...}}
        self.role_probabilities: Dict[str, Dict[str, float]] = {}
        self.my_agent_id: str | None = None
        self.my_role: Role | None = None

    def _get_alive_others(self, status_map: Dict[str, Status]) -> List[str]:
        """自身以外の生存エージェントIDのリストを取得するヘルパー関数。"""
        # Status.ALIVE の値が文字列 'ALIVE' であると仮定
        return [agent_id for agent_id, status in status_map.items() 
                if status == Status.ALIVE and agent_id != self.my_agent_id]

    def initialize_probabilities(self, my_agent_id: str, my_role: Role, game_setting: Setting, all_agents: List[str]) -> None:
        """
        ゲーム設定に基づき、初期確率（事前確率: Prior）を設定する。
        
        Args:
            my_agent_id (str): 自身のAgentID
            my_role (Role): 自身の役職
            game_setting (Setting): ゲーム設定情報
            all_agents (List[str]): 全参加エージェントのIDリスト (例: ['シズエ', 'トシオ', ...])
        """
        self.my_agent_id = my_agent_id
        self.my_role = my_role
        self.role_probabilities = {}

        # 参加プレイヤーの総数
        total_agents = game_setting.agent_count # 修正: agent_count を使用
        
        # 役職ごとの人数構成マップ (ログの構造と一致)
        role_num_map: Dict[Role, int] = getattr(game_setting, 'role_num_map', {}) 
        
        if not role_num_map or total_agents == 0:
            self.logger.logger.error("Role number map not found or total agents is 0. Initialization failed.")
            return

        # 自身を除く、他のエージェントのIDリストを取得 (修正: 渡されたリストを使用)
        other_agents: List[str] = [agent_id for agent_id in all_agents if agent_id != self.my_agent_id]
        
        # 役職リストをRoleオブジェクトのリストに変換
        all_possible_roles = [r for r in POSSIBLE_ROLES if role_num_map.get(r, 0) > 0]
        
        # 1. 自身の役職以外の、他のプレイヤーが担当しうる役職の数を計算
        available_role_counts: Dict[Role, int] = {}
        for role, count in role_num_map.items():
            if role == my_role:
                available_role_counts[role] = count - 1
            else:
                available_role_counts[role] = count
        
        # 2. 各プレイヤーの初期確率を計算
        num_other_agents = len(other_agents)
        if num_other_agents == 0:
            self.logger.logger.warning("Only one player in the game or failed to get other agents.")
            return

        for agent_id in other_agents:
            self.role_probabilities[agent_id] = {}
            
            # 各役職の初期確率: P(A is R) = (役職Rの残り人数) / (他のプレイヤーの総数)
            for role in all_possible_roles:
                role_name = role.name
                prob = available_role_counts.get(role, 0) / num_other_agents
                self.role_probabilities[agent_id][role_name] = prob

            # 正規化 (合計が1になることを保証)
            current_sum = sum(self.role_probabilities[agent_id].values())
            if current_sum > 0:
                for role_name in self.role_probabilities[agent_id]:
                    self.role_probabilities[agent_id][role_name] /= current_sum
            
        self.logger.logger.info("Belief Model initialized.")

    def update(self, game_info: Info, talk_history: List[Talk], agent_role: Role) -> None:
        """OpenAI APIを呼び出し、モジュールパイプラインに従って応答を取得・解析する。"""
        
        # 1. 観察された新しいイベント E を特定する
        # NOTE: _extract_new_events の実装はトーク解析などが必要なため、暫定的に空のイベントリストを返す
        # 実際の運用では、このメソッドでCOや結果を正確に抽出してください。
        new_events = self._extract_new_events(game_info, talk_history) 
        
        # 自身が生存しているプレイヤーに絞る
        alive_agents = [agent_id for agent_id, status in game_info.status_map.items() if status == Status.ALIVE]

        # 2. 各イベント E に対して更新ループを実行する
        for event in new_events:
            
            # 3. 各プレイヤー (Target) ごとに尤度を計算し、確率を更新
            for target_agent_id in alive_agents:
                
                # 自身は信念モデルの対象外
                if target_agent_id == self.my_agent_id:
                    continue
                    
                # 尤度 L(E | R_hypothesis) の計算 (target_agent_id が各役職 R の場合の尤度)
                likelihoods: Dict[str, float] = self._calculate_likelihood(
                    event, target_agent_id, self.role_probabilities
                )

                # 4. ベイズ更新 P(R | E)
                new_probabilities: Dict[str, float] = {}
                total_likelihood_of_E = 0.0 # P(E) の計算 (正規化項)

                for role_name, prior_prob in self.role_probabilities[target_agent_id].items():
                    likelihood = likelihoods.get(role_name, 1.0) # 尤度が定義されていなければ1.0（影響なし）
                    
                    # 事後確率の非正規化項 = 事前確率 * 尤度
                    unnormalized_posterior = prior_prob * likelihood
                    
                    new_probabilities[role_name] = unnormalized_posterior
                    total_likelihood_of_E += unnormalized_posterior
                    
                # 5. 正規化と確率の適用
                if total_likelihood_of_E > 0:
                    for role_name in new_probabilities:
                        # P(R | E) = (P(R) * L(E|R)) / P(E)
                        self.role_probabilities[target_agent_id][role_name] = new_probabilities[role_name] / total_likelihood_of_E
                        
            self.logger.logger.debug(f"Belief Model updated after event: {event['type']}")

    def _extract_new_events(self, game_info: Info, talk_history: List[Talk]) -> List[Dict[str, Any]]:
        """ゲーム情報から新しいCOや結果のイベントを抽出する。"""
        events = []
        
        # 1. COイベント (Talkから抽出) - 全履歴をスキャンし、COパターンを検出
        # (実装省略: Talkのテキスト解析が必要。例: "CO SEER"や"私は占い師です"など)
        # ここでは、CO Talkを検出するシンプルなロジックを想定します。
        # 例: if "占い師CO" in talk.text: events.append({...})
        
        # 2. 占い/霊媒結果イベント (Judgeオブジェクトから抽出)
        
        # 占い結果履歴 (Agent.pyのdivine_results_history構造を利用)
        # NOTE: Agentクラス側で結果をリストに蓄積しているため、そちらを参照する必要がありますが、
        # BeliefModelにアクセス権がないため、一旦Infoパケットの `divine_result` を基に処理します。
        
        current_divine: Judge | None = game_info.divine_result
        if current_divine and current_divine.day == game_info.day - 1:
            # 前日夜に占いが行われた場合
            events.append({
                'type': 'DIVINE_JUDGE',
                'day': game_info.day - 1,
                'agent': current_divine.agent, # 占い師のID (Agent.pyからJudgeを渡す必要あり)
                'target': current_divine.target,
                'result': current_divine.result, # Role or Species
            })

        # 3. 追放イベント (Executed)
        if game_info.executed_agent:
            events.append({
                'type': 'EXECUTION',
                'day': game_info.day - 1, # 追放は前日の議論の結果
                'agent': game_info.executed_agent,
                'target': None,
                'result': None,
            })

        # ... 他のイベント（襲撃, 投票内容の集計など）も同様に追加 ...
        
        return events


    def _calculate_likelihood(self, event: Dict[str, Any], target_agent_id: str, current_probs: Dict) -> Dict[str, float]:
        """
        イベント E が発生したとき、target_agent_id の各役職仮説に対する尤度 L(E | R) を計算する。
        """
        likelihoods = {}
        
        for role_name in self.role_probabilities[target_agent_id]:
            role = Role[role_name] # 仮説 R
            L = 0.0 # 尤度 L(E | R)

            match event['type']:
                
                case 'EXECUTION':
                    # 追放されたのが自分自身の場合、その役職である尤度を計算
                    if target_agent_id == event['agent']:
                        # Rが人狼だった場合 (人狼は追放されやすい)
                        if role == Role.WEREWOLF:
                            L = 0.8 
                        # Rが村人だった場合 (村人は追放されにくい)
                        elif role.is_human():
                            L = 0.2
                        # (これらはあくまで例です。ロジックはより洗練されるべきです)
                        
                case 'CO':
                    # イベント主体が自分自身（target_agent_id）の場合の尤度
                    if target_agent_id == event['agent']:
                        # event['result'] はCOされた役職
                        R_co = event['result']
                        
                        if role == R_co:
                            # Rが真の役職者: L = P_TRUE
                            L = self.P_TRUE
                        elif role == Role.WEREWOLF:
                            # Rが人狼: L = P_WOLF_LIE (偽COする尤度)
                            # R_co が SEER, MEDIUM の場合は高く、VILLAGER の場合は低くするなどの戦略的調整が必要
                            if R_co in [Role.SEER, Role.BODYGUARD]:
                                L = self.P_WOLF_LIE 
                            else:
                                L = self.P_WOLF_NO_LIE # 意味のないCO
                        elif role == Role.POSSESSED:
                            # Rが狂人: L = P_WOLF_LIE (人狼陣営有利なCOをする尤度)
                            if R_co in [Role.SEER, Role.MEDIUM]:
                                L = self.P_WOLF_LIE 
                            else:
                                L = self.P_WOLF_NO_LIE
                        elif role.is_human():
                            # Rが村人陣営: L = P_FALSE_CO (誤ってCOするノイズ)
                            L = self.P_FALSE_CO

                case 'DIVINE_JUDGE':
                    # ここでは、CO者が結果を発表した場合、CO者にCO役職の確信度を割り当て、
                    # 対象者(target_agent_id)に結果の尤度を割り当てる。
                    
                    # 占い師Sが T を R_result と判定したイベント
                    S = event['agent'] 
                    R_result = event['result']
                    
                    # 尤度を計算するのは、Sの役職に関する仮説 L(E | S_is_R)
                    if target_agent_id == S:
                        # Rが真の占い師だった場合 L = P_TRUE (結果を公表する尤度)
                        if role == Role.SEER:
                            L = self.P_TRUE
                        # Rが偽の占い師だった場合 (人狼/狂人) L = P_WOLF_LIE (戦略的な結果を出す尤度)
                        elif role in [Role.WEREWOLF, Role.POSSESSED]:
                            L = self.P_WOLF_LIE
                        # Rがその他の役職だった場合 L = P_FALSE_CO (他人のCOを勝手に報告するノイズ)
                        else:
                            L = self.P_FALSE_CO
                            
                    # 尤度を計算するのは、Tの役職に関する仮説 L(E | T_is_R)
                    elif target_agent_id == event['target']:
                        # Tの役職が R_result と一致する場合 (占い結果と一致)
                        if role == R_result: 
                            # Sが真の占い師である確率 (current_probs[S][SEER]) を尤度として使用
                            L = current_probs[S][Role.SEER.name] 
                        # Tの役職が R_result と一致しない場合
                        else:
                            # 1 - (Sが真の占い師である確率) を尤度として使用
                            L = 1.0 - current_probs[S][Role.SEER.name] 
                    
            # どのイベントにも当てはまらない、または尤度が未定義の場合のデフォルト値
            likelihoods[role_name] = L if L > 0 else 1.0 # 影響を与えない場合は1.0（中立）
            
        return likelihoods

    def get_top_beliefs_summary(self) -> str:
        """
        現在の役職推定結果のサマリー（プロンプト用）を返す。
        全プレイヤーの全役職確率を整形して出力する。
        """
        if not self.role_probabilities:
            return "役職確率はまだ計算されていません。"

        summary_lines = ["--- 全プレイヤーの役職確率推定 (Belief Model) ---"]
        
        # 確率が高い順に役職名を取得するためのユーティリティ関数
        def _format_roles(probabilities: Dict[str, float]) -> str:
            # 確率を降順にソートし、パーセンテージ形式に整形
            sorted_roles = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            # 少数第1位まで表示
            formatted_list = [f"{role}: {prob:.1%}" for role, prob in sorted_roles if prob > 0.0]
            return ", ".join(formatted_list)

        # プレイヤーごとに整形
        for agent_id, role_probs in self.role_probabilities.items():
            if agent_id == self.my_agent_id:
                continue
            
            formatted_probs = _format_roles(role_probs)
            summary_lines.append(f"- {agent_id}: {formatted_probs}")
            
        return "\n".join(summary_lines)

    def decide_target_for_vote(self, status_map: Dict[str, Status]) -> str | None:
        """
        投票ターゲットを決定する。
        原則として、人狼確率が最も高いプレイヤーを選択する（発言との一貫性を重視）。
        """
        alive_others = self._get_alive_others(status_map)
        if not alive_others:
            return None

        # 投票対象は「最も人狼確率が高いプレイヤー」とする。
        # 人狼陣営の場合も、村人陣営を欺くために、最も怪しいプレイヤーに投票する（という発言をする）のが自然であるため、
        # 戦略的な複雑な投票ロジックはLLMの発言に委ねる前提で、行動決定ロジックはシンプルに保つ。
        
        max_wolf_prob = -1.0
        target = None
        
        for agent_id in alive_others:
            # P(WEREWOLF) の確率を取得 (ない場合は 0.0)
            prob_wolf = self.role_probabilities.get(agent_id, {}).get(Role.WEREWOLF.name, 0.0)
            
            if prob_wolf > max_wolf_prob:
                max_wolf_prob = prob_wolf
                target = agent_id
                
        return target or alive_others[0] # フォールバックとして最初の生存者に投票
    
    
    def decide_target_for_divine(self, status_map: Dict[str, Status]) -> str | None:
        """
        占いターゲットを決定する。
        占い師（SEER）のみが実行。情報価値の最大化を狙う。
        """
        if self.my_role != Role.SEER:
            return None 
            
        alive_others = self._get_alive_others(status_map)
        if not alive_others:
            return None

        # 占い師: 情報価値の高いプレイヤーを占う
        # シンプルに、最も人狼確率が高いプレイヤーを占う (黒を確定させる目的)
        max_prob = -1.0
        target = None
        
        for agent_id in alive_others:
            prob_wolf = self.role_probabilities.get(agent_id, {}).get(Role.WEREWOLF.name, 0.0)
            
            # TODO: 既に占ったプレイヤーを除外するロジックを Agent.py から情報を受け取って実装すべき
            
            if prob_wolf > max_prob:
                max_prob = prob_wolf
                target = agent_id
                
        return target or alive_others[0] # フォールバック
        
        
    def decide_target_for_attack(self, status_map: Dict[str, Status]) -> str | None:
        """
        襲撃ターゲットを決定する。
        人狼（WEREWOLF）のみが実行。村人陣営の脅威となる役職を排除する。
        """
        if self.my_role != Role.WEREWOLF:
            return None 
            
        alive_others = self._get_alive_others(status_map)
        if not alive_others:
            return None
        
        # 人狼: 真の役職者（SEER, MEDIUM, BODYGUARD）の確率の合計が最も高いプレイヤーを狙う
        max_threat_prob = -1.0
        target = None
        
        for agent_id in alive_others:
            probs = self.role_probabilities.get(agent_id, {})
            # 脅威度 = P(SEER) + P(MEDIUM) + P(BODYGUARD) の合計
            prob_threat = probs.get(Role.SEER.name, 0.0) + \
                          probs.get(Role.MEDIUM.name, 0.0) + \
                          probs.get(Role.BODYGUARD.name, 0.0) 
                          
            # NOTE: 人狼仲間（POSSESSED）は除外しない（通常、POSSESSEDは襲撃対象ではないため確率は低い）
            # ここでは、人狼仲間を襲撃する確率は極めて低い（0に近い）と仮定し、上記のロジックを適用
            
            if prob_threat > max_threat_prob:
                max_threat_prob = prob_threat
                target = agent_id
                
        return target or alive_others[0] # フォールバック