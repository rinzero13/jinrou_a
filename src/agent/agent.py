"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

import os
from openai import OpenAI, APIError

# --- 新規モジュールのインポート ---
from modules.policy_generator import UtterancePolicyGenerator
from modules.lie_strategist import LieStrategyModule
from modules.consistency_checker import LogicalConsistencyChecker

from modules.belief_model import BeliefModel 

import json

LLM_MODEL = "gpt-4o-mini"  # 発話生成用のモデル

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        # ログファイルパスの設定
        # AgentLoggerが self.log_dir 属性を持つことを前提とする
        if hasattr(self.agent_logger, 'log_dir'):
            log_dir = self.agent_logger.log_dir
            # Strategy Log Pathの設定
            self.strategy_log_path = log_dir / f"strategy_decision_{name}_{game_id}.jsonl"
            
            # ディレクトリが存在しない場合は作成（AgentLogger側で既に作成済みだが念のため）
            # log_dir.mkdir(parents=True, exist_ok=True) # AgentLogger側で作成済みのため省略可能
            self.agent_logger.logger.info(f"Strategy log path: {self.strategy_log_path}")
        else:
            # AgentLoggerがファイル出力を無効化している場合、戦略ロギングを無効化
            self.strategy_log_path = None
            self.agent_logger.logger.warning("AgentLogger did not initialize a log directory. Strategy logging disabled.")
        # ---------------------------------------------
        # ---------------------------------------------

        # --- OpenAIクライアントの初期化 ---
        self.openai_client: OpenAI | None = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                # 環境変数が設定されていても初期化に失敗した場合
                self.agent_logger.logger.error("Failed to initialize OpenAI client: %s", e)
        else:
            # 環境変数が設定されていない場合、ログで警告
            self.agent_logger.logger.warning("OPENAI_API_KEY environment variable is not set. Using random talk mode.")
        # ---------------------------------------------
        
        # --- カスタムモジュールの設定と初期化 ---
        module_settings = self.config.get("custom_modules", {})
        
        self.USE_M3_POLICY = module_settings.get("enable_module_policy", True)
        self.USE_M2_LIE = module_settings.get("enable_module_lie", False)
        self.USE_M1_CONSISTENCY = module_settings.get("enable_module_consistency", False)
        self.MAX_REGENERATION_ATTEMPTS = module_settings.get("max_regeneration_attempts", 2)

        # モジュールインスタンスの作成
        self.M3_Policy = UtterancePolicyGenerator(self.agent_logger) if self.USE_M3_POLICY else None
        # M2はM3を前提とするが、ここでは独立したフラグで制御
        self.M2_Lie = LieStrategyModule(self.agent_logger) if self.USE_M2_LIE else None
        self.M1_Consistency = LogicalConsistencyChecker(self.agent_logger, self.openai_client) if self.USE_M1_CONSISTENCY and self.openai_client else None
        
        # ---------------------------------------------

        # --- 信念モデルの初期化 ---
        self.BeliefModel = BeliefModel(self.agent_logger) 
        # ---------------------------------------------

        # 0日目に発言済みかどうかのフラグ
        self.has_talked_on_day0 = False
        # ---------------------------------------------

        # ---占い/霊媒結果の履歴を保持するリストを初期化 ---
        self.divine_results_history: list[dict[str, Any]] = []
        self.medium_results_history: list[dict[str, Any]] = []
        # -----------------------------------------------------------------

        self.comments: list[str] = []
        with Path.open(
            Path(str(self.config["path"]["random_talk"])),
            encoding="utf-8",
        ) as f:
            self.comments = f.read().splitlines()

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            #占い結果/霊媒結果履歴のリセット
            self.divine_results_history = []
            self.medium_results_history = []

        # --- 履歴の蓄積ロジック ---
        if self.info and self.info.day > 0:
            
            # --- 占い結果の保存 ---
            current_divine_result = getattr(self.info, 'divine_result', None)
            
            # Judgeオブジェクト（または類似の属性を持つオブジェクト）が存在する場合
            if current_divine_result is not None and hasattr(current_divine_result, 'day'):
                # 履歴に保存するための辞書形式に変換し、属性を参照
                result_dict = {
                    # day, agent, targetなどの属性をgetattrで安全に取得
                    'day': getattr(current_divine_result, 'day', None),
                    'agent': getattr(current_divine_result, 'agent', None),
                    'target': getattr(current_divine_result, 'target', '不明'),
                    # resultはRoleやSpeciesオブジェクトの可能性が高いため、.name属性があればそれを、なければ文字列化を試みる
                    'result': getattr(getattr(current_divine_result, 'result', '不明'), 'name', str(getattr(current_divine_result, 'result', '不明'))),
                }
                
                # 最新の結果（前日実施）であり、まだ履歴に記録されていない場合のみ追加
                if (
                    result_dict.get('day') == self.info.day - 1
                    and not any(r.get('day') == result_dict.get('day') for r in self.divine_results_history)
                ):
                    self.divine_results_history.append(result_dict)
            
            # --- 霊媒結果の保存（同様のJudgeオブジェクト構造を想定） ---
            current_medium_result = getattr(self.info, 'medium_result', None)
            
            if current_medium_result is not None and hasattr(current_medium_result, 'day'):
                result_dict = {
                    'day': getattr(current_medium_result, 'day', None),
                    'agent': getattr(current_medium_result, 'agent', None),
                    'target': getattr(current_medium_result, 'target', '不明'),
                    'result': getattr(getattr(current_medium_result, 'result', '不明'), 'name', str(getattr(current_medium_result, 'result', '不明'))),
                }
                
                if (
                    result_dict.get('day') == self.info.day - 1
                    and not any(r.get('day') == result_dict.get('day') for r in self.medium_results_history)
                ):
                    self.medium_results_history.append(result_dict)
        # -----------------------------------------------------------

        # --- 信念モデルの更新 ---
        if self.info and self.BeliefModel: 
            # 最新のゲーム情報、全発話履歴、自身の役職を渡して確率を更新
            self.BeliefModel.update(self.info, self.talk_history, self.role)
        # ----------------------------

        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        # ゲーム開始時に０日目の発言フラグをリセット
        self.has_talked_on_day0 = False

        # --- Belief Modelの初期化メソッド呼び出し ---
        if self.BeliefModel and self.setting and self.info:
            # Info.status_map のキー（キャラクター名）が全 Agent ID
            # Info.status_mapには全Agentの状態が含まれるため、これをそのまま使う
            all_agents = list(self.info.status_map.keys()) 
            self.BeliefModel.initialize_probabilities(
                my_agent_id=self.agent_name, 
                my_role=self.role, 
                game_setting=self.setting,
                all_agents=all_agents # 全エージェントのリストを渡す
            )
        # ----------------------------------------------------

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        return random.choice(self.comments)  # noqa: S311

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        # --- 0日目の発言制限ロジック (2ターン目以降「Over」) ---
        # self.infoが存在し、かつ0日目の場合
        if self.info and self.info.day == 0:
            if not self.has_talked_on_day0:
                # 0日目・初回の発言は許可し、フラグを立てる
                self.has_talked_on_day0 = True
                # LLMの発言生成ロジックへ
            else:
                # 0日目・2回目以降の発言はOverを返し、その日の発言を終了
                self.agent_logger.logger.info("Day 0, subsequent talk requested. Returning Over.")
                return "Over" # その日の発言フェーズを終了するメッセージ
        # --- 0日目の発言制限ロジック ---


        # --- OpenAI APIを呼び出すロジック ---
        if self.openai_client and self.info: # self.info (ゲーム情報) がセットされているか確認
            llm_response = self._call_openai_api()
            if llm_response:
                return llm_response
                
        # API呼び出しに失敗した場合やクライアントがない場合は、従来のランダム発言をフォールバックとして使用
        return random.choice(self.comments)  # noqa: S311

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """

        if self.BeliefModel and self.info:
            target = self.BeliefModel.decide_target_for_divine(self.info.status_map) 
            if target:
                return target
            
        return random.choice(self.get_alive_agents())  # noqa: S311

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """

        if self.BeliefModel and self.info:
            target = self.BeliefModel.decide_target_for_vote(self.info.status_map) 
            if target:
                return target
            
        return random.choice(self.get_alive_agents())  # noqa: S311

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """

        if self.BeliefModel and self.info:
            target = self.BeliefModel.decide_target_for_attack(self.info.status_map) 
            if target:
                return target
            
        return random.choice(self.get_alive_agents())  # noqa: S311

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    # 発話生成用のプロンプト生成
    def _create_talk_prompt(self, m3_plan: dict, m2_strategy: dict, regeneration_feedback: str = None) -> tuple[str, str]:
            """
            LLMへのシステムメッセージとユーザープロンプトを生成する。
            M3とM2の指示を統合し、M1からの矛盾修正指示（regeneration_feedback）を組み込む。
            
            Args:
                regeneration_feedback (str): M1からの矛盾修正指示（ある場合、NoneでOK）
            
            Returns:
                tuple[str, str]: (system_message, user_prompt)
            """
            if not self.info:
                # ゲーム情報がない場合のフォールバック
                return "system_message", f"ゲーム情報がありません。ランダムな発言をします。エージェント名: {self.agent_name}" 

            # ----------------------------------------------------
            # 1. 基本情報の取得
            # ----------------------------------------------------
            agent_display_name = self._get_agent_display_name()
            
            # ----------------------------------------------------
            # 2. システムメッセージ (System Message) の構築
            # ----------------------------------------------------
            system_message = (
                f"目的は役職の勝利条件を達成することです。\n"
                f"**【重要】発言は**`【発言：】`**の直後に記述し、その内容は**厳密に120文字以内**の簡潔な日本語で、あなたの役職が疑われないように行ってください。\n" 
                f"**最終的な出力の形式を厳守してください。発言内容の直前には必ず**`【発言：】`**というラベルを付けてください。**`【発言：】`**より前の思考プロセスは、ゲーム内で発言されません。あなたの**最終的な発言**は、`【発言：】`ラベルの直後の内容のみが採用されます。\n"  
            )
            
            # --- 役職ごとの具体的な戦略的指示---
            match self.role:
                case Role.WEREWOLF:
                    system_message += "あなたは人狼です。正体を隠し、村人陣営の疑いをそらすことが最重要です。他のプレイヤーへの協力を装いつつ、村人同士の対立を煽り、議論を有利な方向へ誘導してください。特定のプレイヤーの発言の矛盾や信頼性を具体的に指摘しなさい。\n"
                case Role.POSSESSED:
                    system_message += "あなたは狂人です。人狼の勝利のために、村人陣営の議論を混乱させ、誤ったプレイヤー（村人陣営）に投票が集まるよう誘導してください。人狼の味方となる発言や、人狼を白く見せるような発言を心がけなさい。\n"
                case Role.SEER:
                    system_message += "あなたは占い師です。得られた情報（議論や占い結果）を基に、信頼性を高めつつ、村人陣営を勝利に導くために最も合理的なプレイヤーを指摘してください。議論をリードしなさい。\n"
                case _: # VILLAGER, BODYGUARD, MEDIUM, etc.
                    system_message += "あなたは村人陣営です。生存者の発言を注意深く分析し、論理的推論に基づき怪しいプレイヤーを指摘するか、他のプレイヤーと協力し、村人陣営の勝利を目指してください。\n"

            # --- M1: 論理矛盾修正指示の組み込み（再生成ループ時のみ） ---
            if regeneration_feedback:
                system_message += f"\n\n**【論理矛盾の修正指示】**\n"
                system_message += f"あなたの前回の発言は以下の点で論理的に矛盾していると判定されました。**この修正指示を完全に反映し、矛盾を解消した発言を再生成してください。**\n"
                system_message += f"修正理由: {regeneration_feedback}\n"
                self.agent_logger.logger.warning("M1 feedback integrated into system prompt for regeneration.")

            # ----------------------------------------------------
            # 3. ユーザープロンプト (User Prompt) の構築
            # ----------------------------------------------------
            
            game_state_summary = self._summarize_game_state()
            formatted_history = self._format_talk_history()
            
            # M3/M2の結果をUser Promptのコンテキストとして追加
            strategy_context = ""
            
            # M3またはM2のプランが存在する場合のみコンテキストを生成する
            if m3_plan or m2_strategy:
                # M3情報
                m3_core_goal = m3_plan.get('core_goal', '情報収集に徹する')
                m3_response_policy = m3_plan.get('response_policy', 'PRIORITIZE_CORE')
                
                # M2情報 (M2が無効で空のdictの場合、extended_goalはcore_goalと同じになる)
                m2_extended_goal = m2_strategy.get('extended_goal', m3_core_goal)
                m2_lie_used = 'はい' if m2_strategy.get('lie_used') else 'いいえ'

                # モジュール使用時のみ表示されるコンテキスト
                strategy_context = (
                    "\n----------------------------------------------------\n"
                    "**【事前決定された戦略的目標と方針】**\n"
                    f"**1. 主張の核 (Core Goal):** {m3_core_goal}\n"
                    f"**2. 議論への対応方針 (Response Policy):** {m3_response_policy} (RESPOND_CRITICALLYの場合、直前の発言への応答を最優先してください)\n"
                    f"**3. 嘘戦略の使用:** {m2_lie_used}\n"
                    f"**4. 最終的な行動目標:** {m2_extended_goal} (これがあなたの発言の目標です)\n"
                    "----------------------------------------------------\n"
                )

            user_prompt = f"""
                あなたは {agent_display_name} です。

                【現在のゲーム情報】
                {game_state_summary}

                {strategy_context}

                【これまでの会話履歴（直近10件）】
                {formatted_history}

                【あなたの発言決定プロセス】
                情報を踏まえて、目標達成のために発言内容を120字以内で決定してください。
            """
            return system_message, user_prompt
    # ----------------------------------------------------------------------
    # Agent ID と Agent Name のマッピングに関するヘルパー関数
    # ----------------------------------------------------------------------

    def _get_agent_display_name(self) -> str:
            """現在のエージェントの表示名（シオン、ベンジャミンなど）を取得する。"""
            # Infoパケットの agent フィールド（表示名）を直接利用
            if self.info and hasattr(self.info, 'agent'):
                return self.info.agent
            # 取得できない場合は、初期化時のエージェント名（コネクタID）をフォールバックとして使用
            return self.agent_name


    def _get_role_knowledge(self) -> str:
        """役職ごとの知っている情報を取得する。（履歴リストを参照するように修正）"""
        if not self.info:
            return "初期情報なし。"

        match self.role:
            case Role.WEREWOLF:
                wolf_list = getattr(self.info, 'werewolf_agent_list', [])
                if not isinstance(wolf_list, list):
                    wolf_list = []
                wolf_list = [a for a in wolf_list if a != self.agent_name]
                return f"人狼仲間: {', '.join(wolf_list)}、勝利条件: 村人陣営の数を人狼陣営の数以下にする。"
            
            case Role.SEER:
                results = []
                
                # 蓄積された履歴リストから結果を取得し整形
                for result in self.divine_results_history:
                    day = result.get('day', '?')
                    target = result.get('target', '不明')
                    role_str = result.get('result', '不明') 
                    
                    results.append(f"Day{day} {target} -> {role_str}")

                return f"これまでの占い結果: {', '.join(results) if results else 'まだ占い結果は出ていません。'}。"

            case Role.MEDIUM:
                results = []
                
                # 蓄積された履歴リストから結果を取得し整形
                for result in self.medium_results_history:
                    day = result.get('day', '?')
                    target = result.get('target', '不明')
                    role_str = result.get('result', '不明')
                    
                    results.append(f"Day{day} {target} (追放) -> {role_str}")
                
                return f"これまでの霊媒結果: {', '.join(results) if results else 'まだ霊媒結果は出ていません。'}。"

            case Role.BODYGUARD:
                guarded_agent = getattr(self.info, 'guarded_agent', 'なし')
                if guarded_agent is None:
                    guarded_agent = 'なし'
                return f"護衛結果（前日夜）：{guarded_agent}。"
            
            case _:
                return "特に確定情報はありません。"
            
    def _summarize_game_state(self) -> str:
        """現在のゲーム状況をサマリーする。"""
        if not self.info: 
            return "ゲーム情報なし。"
        
        # status_map から生存エージェントの名前を取得
        alive_agent_names = [k for k, v in self.info.status_map.items() if v == 'ALIVE']
        
        executed = self.info.executed_agent if self.info.executed_agent else "なし"
        attacked = self.info.attacked_agent if self.info.attacked_agent else "なし"
        
        belief_summary = ""
        if self.BeliefModel:
            belief_summary = self.BeliefModel.get_top_beliefs_summary() # BeliefModelの要約メソッドを呼び出す
        
        return (
            f"日目: Day {self.info.day}\n"
            f"生存者: {', '.join(alive_agent_names)}\n"
            f"前回追放: {executed}\n"
            f"前回襲撃: {attacked}\n"
            f"あなたの役職の知っていること: {self._get_role_knowledge()}"
            f"\n--- 信念モデルによる推定 ---\n"
            f"{belief_summary}" # LLMへのプロンプトに含める
        )

    def _format_talk_history(self, limit: int = 10) -> str: # limit引数を追加し、デフォルト値を設定
        """会話履歴を整形する。オプションのlimitで表示件数を制限する。"""
        formatted = []
        
        # ADD LINE: Day 0の発言を除外した履歴リストを作成
        # 0日目の発言は論理的情報として利用しない
        filtered_history = [talk for talk in self.talk_history if talk.day != 0]

        # limitに基づいて履歴をスライス
        # デフォルトの10件、または指定された件数（例: M3の1件）に制限される
        history_to_format = filtered_history[-limit:]


        for talk in history_to_format:
            speaker = talk.agent 
            formatted.append(f"D{talk.day} {speaker}: {talk.text}")
            
        return "\n".join(formatted) if formatted else "まだ会話はありません。"
    
    # ----------------------------------------------------------------------
    def _log_strategy_decision(self, m3_plan: dict, m2_strategy: dict, strategy_log: str, final_talk: str, system_prompt: str, user_prompt: str) -> None: # <-- 修正: プロンプト引数を追加
            """決定された戦略、LLMの思考プロセス、および最終発言をJSONLファイルに記録する。"""
            
            if not self.strategy_log_path: # このチェックで log_dir がNoneの場合のエラーを防ぐ
                self.agent_logger.logger.warning("Strategy log path is not set. Skipping log decision.")
                return

            # M3/M2の結果は、モジュールが無効の場合に空の辞書 {} となるため、適切にログに記録する。
            
            log_entry = {
                "day": self.info.day if self.info else 0,
                "role": self.role.name,
                "is_m3_used": bool(self.USE_M3_POLICY),
                "is_m2_used": bool(self.USE_M2_LIE),
                "m3_plan": m3_plan if self.USE_M3_POLICY else {},
                "m2_strategy": m2_strategy if self.USE_M2_LIE else {},
                "llm_strategy_log": strategy_log,      # 【発言：】より前のLLMの思考
                "final_talk": final_talk,              # 最終発言
                # プロンプト全体をログに記録
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            
            try:
                # self.strategy_log_path を使用 (__init__で定義済み)
                with self.strategy_log_path.open(mode="a", encoding="utf-8") as f: 
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                self.agent_logger.logger.debug("Strategy decision logged.")
            except Exception as e:
                self.agent_logger.logger.error(f"Failed to write strategy log file: {e}")
    # ----------------------------------------------------------------------   
    def _safe_json_parse(self, content: str, fallback_data: dict) -> dict:
        """LLM応答からJSONをロバストにパースする。"""
        try:
            # LLMがJSONの前後に説明文などを付加する場合を考慮し、最初の '{' から最後の '}' までを抽出
            json_start = content.find('{')
            json_end = content.rfind('}')
            if json_start == -1 or json_end == -1:
                self.agent_logger.logger.warning(f"JSON marker not found in response: {content[:50]}...")
                return fallback_data
            
            json_str = content[json_start:json_end+1]
            return json.loads(json_str)
        except Exception as e:
            self.agent_logger.logger.error(f"JSON parsing failed. Error: {e}. Content: {content[:50]}...")
            return fallback_data


    def _query_llm_for_strategy(self, prompt_context: str, fallback_data: dict) -> dict:
        """
        戦略決定（M3/M2）専用のLLMコールを行う。
        安定したJSON出力を得るため、専用のSystem Messageと低温度設定を使用する。
        """
        if not self.openai_client:
            self.agent_logger.logger.error("OpenAI client not initialized. Cannot query strategy LLM.")
            return fallback_data

        # JSON出力を強制するためのSystem Message
        system_msg = (
            "あなたは与えられた情報に基づき、戦略的な計画をJSON形式で出力する専門のAIです。\n"
            "思考プロセス、説明、Markdown形式（```json```）など、**JSONオブジェクト以外の一切のテキスト**は含めないでください。**厳密にJSONオブジェクトのみ**を出力してください。"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL, # 最終発話とモデルを分けることも可能
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt_context},
                ],
                temperature=0.2, # 創造性を抑え、一貫性を重視
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            return self._safe_json_parse(content, fallback_data)
            
        except Exception as e:
            self.agent_logger.logger.error(f"Strategy LLM API call failed: {e}")
            return fallback_data


    def _get_m3_plan_from_llm(self) -> dict:
        """M3: 主張の核と応答方針をLLMに問い合わせて決定する。（モジュール不使用時は空の辞書を返す）"""
        fallback_data = {"core_goal": "状況を注意深く観察する", "response_policy": "PRIORITIZE_CORE", "response_target_id": "NONE"}
        
        if not self.USE_M3_POLICY or not self.M3_Policy:
            self.agent_logger.logger.info("M3 Policy Generator is disabled. Using default plan.")
            return {} # 空の辞書を返し、_create_talk_promptでデフォルト処理される

        self.agent_logger.logger.debug("M3: Planning LLM call initiated.")
        
        # M3専用のプロンプトを構築
        m3_instructions = self.M3_Policy.get_planning_prompt_instructions(self.role)
        prompt_context = f"""
        {m3_instructions}

        【現在のゲーム情報】
        {self._summarize_game_state()}
        【直前の発話】
        {self._format_talk_history(limit=1)}
        """
        
        return self._query_llm_for_strategy(prompt_context, fallback_data)


    #### B. M2: 嘘戦略の決定
    # M2_Lie は LieStrategyModule のインスタンスを想定

    def _get_m2_strategy_from_llm(self, m3_plan: dict) -> dict:
        """M2: M3の計画に基づき、嘘戦略の有無を決定する。（モジュール不使用時は空の辞書を返す）"""
        
        # M2モジュール不使用の場合
        if not self.USE_M2_LIE or not self.M2_Lie:
            return {}
        
        core_goal = m3_plan.get('core_goal', '情報収集に徹する')
        fallback_data = {"lie_used": False, "extended_goal": core_goal}

        self.agent_logger.logger.debug("M2: Lie strategy LLM call initiated.")
        
        # M2専用のプロンプトを構築
        m2_instructions = self.M2_Lie.get_strategy_decision_instructions(self.role)
        prompt_context = f"""
        {m2_instructions}

        【あなたの役職情報】
        役職: {self.role.name} / 知っている情報: {self._get_role_knowledge()}
        【M3決定された主張の核】
        {core_goal}
        """
        
        return self._query_llm_for_strategy(prompt_context, fallback_data)
    # ---------------------------------------------
    def _call_openai_api(self) -> str | None:
            """OpenAI APIを呼び出し、モジュールパイプラインに従って応答を取得・解析する。"""
            if not self.openai_client:
                return None
            
            # 【M3/M2の事前戦略決定】
            # -------------------------------------------------------------------
            # モジュールが有効な場合はLLMコールを行い、無効な場合は空の辞書（{}）を取得する。
            m3_plan = self._get_m3_plan_from_llm() 
            m2_strategy = self._get_m2_strategy_from_llm(m3_plan)

            # 【M1: 論理的一貫性判定・修正のためのループ】
            feedback = None
            last_generated_talk = None
            current_system_message = ""
            current_user_prompt = ""
            final_strategy_log = "" 
            
            for attempt in range(self.MAX_REGENERATION_ATTEMPTS):
                # 1. LLMへのプロンプト生成 (M3/M2の指示を統合、またはM1のフィードバックを組み込む)
                system_message, user_prompt = self._create_talk_prompt(
                            m3_plan=m3_plan, 
                            m2_strategy=m2_strategy, 
                            regeneration_feedback=feedback
                        )
                current_system_message = system_message 
                current_user_prompt = user_prompt

                try:
                    # 2. LLM APIコール (仮想発話の生成)
                    response = self.openai_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=350,
                        temperature=0.7,
                    )
                    
                    talk_content_full = response.choices[0].message.content.strip()
                    talk_content = None
                    
                    # 3. 応答の解析と抽出（「発言：」のラベルを基に）
                    if "【発言：】" in talk_content_full:
                        parts = talk_content_full.split("【発言：】", 1)
                        strategy_log = parts[0].strip()
                        talk_content = parts[1].strip()
                        
                        if len(talk_content) > 125:
                            talk_content = talk_content[:125]

                        last_generated_talk = talk_content
                        final_strategy_log = strategy_log

                        self.agent_logger.logger.info(f"Attempt {attempt+1}: LLM Strategy (Internal COG): {strategy_log[:50]}...")
                        self.agent_logger.logger.info(f"Attempt {attempt+1}: LLM Response (Virtual Talk): {talk_content}")

                        # 4. M1: 論理的一貫性チェックの実行
                        # モジュールが有効化されている場合、チェックを行う
                        if self.USE_M1_CONSISTENCY and self.M1_Consistency:
                            is_consistent, reason = self.M1_Consistency.check(
                                game_info=self.info, 
                                talk_history=self.talk_history, 
                                virtual_talk=talk_content, 
                                agent_name=self.agent_name
                            )
                            
                            if is_consistent:
                                # 矛盾なしと判定されたら、ループを終了して発言を返す
                                self.agent_logger.logger.info("M1: Logical consistency check passed. Talk decided.")
                                # <-- 修正/追加箇所: ログ関数呼び出し
                                self._log_strategy_decision(m3_plan, m2_strategy, final_strategy_log, talk_content, current_system_message, current_user_prompt) 
                                return talk_content
                            else:
                                # 矛盾ありと判定されたら、feedbackを更新し、再生成へ
                                self.agent_logger.logger.warning(f"M1: Logical inconsistency detected (Attempt {attempt+1}/{self.MAX_REGENERATION_ATTEMPTS}). Retrying...")
                                feedback = reason # M1からの修正指示をフィードバックとして設定
                                continue # 次のループへ
                        else:
                            # M1が無効の場合、チェックせずに発言を返す
                            self.agent_logger.logger.info("M1: Consistency check skipped (Module disabled). Talk decided.")
                            # <-- 修正/追加箇所: ログ関数呼び出し
                            self._log_strategy_decision(m3_plan, m2_strategy, final_strategy_log, talk_content, current_system_message, current_user_prompt) 
                            return talk_content

                    # テンプレートに従わなかった場合 (既存コードのフォールバック)
                    else:
                        self.agent_logger.logger.warning("LLM response did not contain the '発言：' tag. Using full content as talk.")
                        talk_content = talk_content_full[:100] if len(talk_content_full) > 100 else talk_content_full
                        
                        last_generated_talk = talk_content
                        final_strategy_log = "" # テンプレート外の場合は思考ログなし

                        # テンプレート外でもM1はチェックすべき
                        if self.USE_M1_CONSISTENCY and self.M1_Consistency:
                            is_consistent, reason = self.M1_Consistency.check(
                                game_info=self.info, 
                                talk_history=self.talk_history, 
                                virtual_talk=talk_content, 
                                agent_name=self.agent_name
                            )
                            if is_consistent:
                                # <-- 修正/追加箇所: ログ関数呼び出し
                                self._log_strategy_decision(m3_plan, m2_strategy, final_strategy_log, talk_content, current_system_message, current_user_prompt) 
                                return talk_content
                            else:
                                feedback = reason
                                continue
                        
                        # ログ関数呼び出し
                        self._log_strategy_decision(m3_plan, m2_strategy, final_strategy_log, talk_content, current_system_message, current_user_prompt) 
                        return talk_content # M1が無効、または最初の試行で矛盾なしの場合

                except APIError as e:
                    self.agent_logger.logger.error("OpenAI API Error: %s", e)
                    break 
                except Exception as e:
                    self.agent_logger.logger.error("An unexpected error occurred during API call: %s", e)
                    break

            # ループを抜けたが有効な発言が得られなかった場合
            self.agent_logger.logger.warning("Failed to generate consistent talk after all attempts. Using last generated talk.")
            
            if last_generated_talk:
                # 最後に生成されたものをログに記録
                self._log_strategy_decision(m3_plan, m2_strategy, final_strategy_log, last_generated_talk, current_system_message, current_user_prompt) 
                return last_generated_talk
            else:
                self.agent_logger.logger.warning("No talk was generated at all. Falling back to random choice.")
                return random.choice(self.comments)

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None