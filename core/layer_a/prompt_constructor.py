from typing import List
from core.layer_d.schema import MemoryPoint
import time

class PromptConstructor:
    """Packages memories to prevent Prompt Confusion via XML tagging."""

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escapes XML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    @staticmethod
    def build(query: str, memories: List[MemoryPoint]) -> str:
        """
        Constructs the structured XML-encapsulated prompt zero-latency f-strings.
        
        Args:
            query: The original user question.
            memories: DARS-selected MemoryPoint lists.
            
        Returns:
            The structured prompt string ready to pass to Brain LLM inference.
        """
        system_context = (
            "<system_context>\n"
            "You are an AI assistant integrating long-term retention via DARS framework.\n"
            "Use the provided memory stream to augment your response accurately.\n"
            "CRITICAL METADATA INSTRUCTION:\n"
            "Each memory contains system_weight and last_accessed attributes. These are historical reliability metrics from the DARS framework.\n"
            "High system_weight (>0.80): This memory has been frequently helpful in past successful tasks.\n"
            "Low system_weight (<0.40): This memory is historically less relevant but may still contain the factual answer.\n"
            "Do NOT mention these scores to the user. Do NOT apologize for using low-weight memories. Use them as grounding facts to answer the <current_user_query> accurately.\n"
            "</system_context>\n"
        )

        memory_xml = ["<memory_stream>"]
        for m in memories:
            # Safely get payload attributes, fallback if not set.
            mid = m.point_id
            u = getattr(m.payload, 'utility', 0.0)
            r = getattr(m.payload, 'recency', 0.0)
            
            # Format elapsed time safely, to relative format if preferred or just standard unix
            # We will just print the exact float logic for now.
            escaped_text = PromptConstructor._escape_xml(m.payload.text_content)
            xml_str = (
                f"    <memory id=\"{mid}\" system_weight=\"{u:.2f}\" last_accessed=\"{r:.0f}\">\n"
                f"        {escaped_text}\n"
                f"    </memory>"
            )
            memory_xml.append(xml_str)
        memory_xml.append("</memory_stream>\n")

        # Join the memory sequence
        memory_section = "\n".join(memory_xml)
        
        escaped_query = PromptConstructor._escape_xml(query)
        user_query_xml = (
            f"<current_user_query>\n"
            f"{escaped_query}\n"
            f"</current_user_query>"
        )

        # Stitch all components together
        return f"{system_context}\n{memory_section}\n{user_query_xml}"
