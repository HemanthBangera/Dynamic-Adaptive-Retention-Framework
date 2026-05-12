import logging
from typing import List

from core.layer_d.schema import MemoryPoint
from core.layer_d.storage import chunk_index_from_tags

logger = logging.getLogger(__name__)


class PromptConstructor:
    """Packages memories to prevent Prompt Confusion via XML tagging."""

    _distillation_queue: List[str] = []

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escapes XML special characters and strips null bytes."""
        text = text.replace("\x00", "")
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    @classmethod
    def build(cls, query: str, memories: List[MemoryPoint]) -> str:
        """
        Constructs the structured XML-encapsulated prompt.
        Memories are ordered by narrative chunk index when ``chunk:n`` tags exist.
        Enforces an expanded character budget and truncates outliers.

        Pure function: does NOT mutate any input MemoryPoint objects.
        Oversized memories are queued for distillation via get_distillation_queue().
        """
        cls._distillation_queue = []

        system_context = (
            "<system_context>\n"
            "You are an AI assistant integrating long-term retention via DARS framework.\n"
            "Use the provided memory stream to augment your response accurately.\n"
            "CRITICAL METADATA INSTRUCTION:\n"
            "Each memory contains system_weight and last_accessed attributes. These are historical reliability metrics from the DARS framework.\n"
            "High system_weight (>0.80): This memory has been frequently helpful in past successful tasks.\n"
            "Low system_weight (<0.40): This memory is historically less relevant but may still contain the factual answer.\n"
            "Do NOT mention these scores to the user. Do NOT apologize for using low-weight memories. Use them as grounding facts to answer the <current_user_query> accurately.\n"
            "Memories are listed in ascending story chunk order when chunk tags are present; prefer later chunks for current world state.\n"
            "</system_context>\n"
        )

        memory_xml = ["<memory_stream>"]

        current_len = len(system_context) + len("<memory_stream>\n") + len("</memory_stream>\n") + len(query) + 50
        MAX_PROMPT_CHARS = 48000

        ordered = sorted(
            memories,
            key=lambda m: (
                chunk_index_from_tags(m.payload.tags)
                if chunk_index_from_tags(m.payload.tags) is not None
                else 10**9,
                m.point_id,
            ),
        )

        for m in ordered:
            mid = m.point_id
            weight = m.dars_score if m.dars_score is not None else getattr(m.payload, 'utility', 0.0)
            r = getattr(m.payload, 'recency', 0.0)
            ci = chunk_index_from_tags(m.payload.tags)
            chunk_attr = f' story_chunk="{ci}"' if ci is not None else ""

            raw_text = getattr(m.payload, 'text_content', "")

            if len(raw_text) > 12000:
                raw_text = raw_text[:12000] + "\n[TRUNCATED FOR BUDGET - SEE ORIGINAL_TEXT_BACKUP]"
                cls._distillation_queue.append(mid)

            escaped_text = cls._escape_xml(raw_text)
            xml_str = (
                f"    <memory id=\"{mid}\" system_weight=\"{weight:.2f}\" last_accessed=\"{r:.0f}\"{chunk_attr}>\n"
                f"        {escaped_text}\n"
                f"    </memory>"
            )

            if current_len + len(xml_str) > MAX_PROMPT_CHARS:
                logger.warning("dars_gateway_context_truncated_total: Skipped adding memory %s due to MAX_PROMPT_CHARS limit.", mid)
                break

            memory_xml.append(xml_str)
            current_len += len(xml_str)

        memory_xml.append("</memory_stream>\n")
        memory_section = "\n".join(memory_xml)

        escaped_query = cls._escape_xml(query)
        user_query_xml = (
            f"<current_user_query>\n"
            f"{escaped_query}\n"
            f"</current_user_query>"
        )

        return f"{system_context}\n{memory_section}\n{user_query_xml}"

    @classmethod
    def get_distillation_queue(cls) -> List[str]:
        """Return point IDs of oversized memories that need compression."""
        return list(cls._distillation_queue)
