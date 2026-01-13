import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from aiwolf_nlp_common.packet import Talk

logger = logging.getLogger(__name__)

class SpeechAnalyzer:
    """
    Module 1: 発話分析
    対話履歴から「事実」と「意図」を構造化データとして抽出する。
    """
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def analyze_latest_talks(self, talks: List[Talk], limit: int = 5) -> List[Dict[str, Any]]:
        """直近の発言を解析し、構造化したリストを返す"""
        if not talks:
            return []

        recent_talks = talks[-limit:]
        formatted_history = "\n".join([f"D{t.day} {t.agent}: {t.text}" for t in recent_talks])

        system_prompt = (
            "あなたは人狼ゲームの専門アナリストです。提供された発話履歴の各発言を分析し、以下のJSONスキーマに従って出力してください。\n\n"
            "## 抽出項目:\n"
            "1. day: 発言があった日 (integer)\n"
            "2. agent: 発言者ID (string)\n"
            "3. target: 対象者ID。言及がない場合は 'NONE' (string)\n"
            "4. fact: 抽出された客観的事実。占い結果や役職COなど (string)\n"
            "5. intent: 発言の意図（発話目的）。以下の分類から最も適切なものを1つ選択してください:\n"
            "   - CO: 自己の正当性の確立/情報公開（自身の役職COや、真/偽の証拠提示により信頼を得る）\n"
            "   - ATTACK: 他者の不審点/矛盾の指摘（特定プレイヤーへの告発を行い、議論の焦点を集める）\n"
            "   - GUIDE: 処刑ターゲットの誘導/決定（特定の投票推奨を行い、村の合意形成を加速させる）\n"
            "   - INQUIRY: 特定の情報収集/場への問いかけ（未確定情報：他者の役職や信念を知るための質問を投げる）\n"
            "   - DEFEND: 味方の擁護/陣営の支援（告発された味方を弁護し、議論を自身に有利な方向へ戻す）\n"
            "   - DISRUPT: 情報撹乱・場全体の混乱（議論の焦点を意図的にずらし、村の判断力を低下させる）\n"
            "   - NONE: 上記のいずれにも該当しない、または明確な意図がない場合\n"
            "6. summary: 30文字以内の簡潔な要約 (string)\n\n"
            "## 出力形式 (JSON List):\n"
            "[\n"
            "  {\n"
            "    \"day\": ,\n"
            "    \"agent\": \"agent_id\",\n"
            "    \"target\": \"target_id\",\n"
            "    \"fact\": \"fact_content\",\n"
            "    \"intent\": \"intent_type\",\n"
            "    \"summary\": \"summary_text\"\n"
            "  }\n"
            "]\n\n"
            "## 制約事項:\n"
            "- 出力は必ず有効なJSONオブジェクトのリストのみとしてください。\n"
            "- 解説やMarkdownのコードブロック(```json)は含めないでください。"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"分析対象の発言:\n{formatted_history}"}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            # JSON部分の抽出とパース
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            return json.loads(content[json_start:json_end])
        except Exception as e:
            logger.error(f"Speech analysis failed: {e}")
            return []