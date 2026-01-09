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

    def analyze_latest_talks(self, talks: List[Talk], limit: int = 3) -> List[Dict[str, Any]]:
        """直近の発言を解析し、構造化したリストを返す"""
        if not talks:
            return []

        recent_talks = talks[-limit:]
        formatted_history = "\n".join([f"D{t.day} {t.agent}: {t.text}" for t in recent_talks])

        system_prompt = (
            "あなたは人狼ゲームの専門アナリストです。各発言から以下の項目を抽出し、JSON形式のリストで出力してください。\n"
            "1. agent: 発言者ID\n"
            "2. target: 対象者ID（不明な場合は 'NONE'）\n"
            "3. fact: 抽出された事実（例：占い結果、役職CO）\n"
            "4. intent: 発言の意図（分類：CO, VOTE_REQUEST, ATTACK, DEFEND, INQUIRY, AGREE, DISAGREE）\n"
            "5. summary: 15文字以内の簡潔な要約\n"
            "出力はJSONのみとし、余計な解説は含めないでください。"
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