import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
} from "@huggingface/transformers";
import { MODEL_CONFIG, CLIP_MODEL_NAME } from "@/transformers/config";

const textProcessor = await AutoTokenizer.from_pretrained(CLIP_MODEL_NAME);
const textModel = await CLIPTextModelWithProjection.from_pretrained(
  CLIP_MODEL_NAME,
  MODEL_CONFIG
);

export default async function generateTextEmbedding(
  text: string
): Promise<number[]> {
  const textInputs = await textProcessor(text);
  const { text_embeds } = await textModel(textInputs);

  return text_embeds.normalize().tolist()[0];
}
