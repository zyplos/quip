import { AutoTokenizer, SiglipTextModel } from "@huggingface/transformers";
import { MODEL_CONFIG, SIGLIP_MODEL_NAME } from "../config";

const textProcessor = await AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME);
const textModel = await SiglipTextModel.from_pretrained(
  SIGLIP_MODEL_NAME,
  MODEL_CONFIG
);

export default async function generateTextEmbedding(
  text: string
): Promise<number[]> {
  const textInputs = await textProcessor(text);
  const { pooler_output } = await textModel(textInputs);

  return pooler_output.normalize().tolist()[0];
}
