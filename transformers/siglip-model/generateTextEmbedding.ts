import { AutoTokenizer, SiglipModel } from "@huggingface/transformers";
import { MODEL_CONFIG, SIGLIP_MODEL_NAME } from "@/transformers/config";
import { createEmptyTensor } from "../utils";

const textProcessor = await AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME);
const model = await SiglipModel.from_pretrained(
  SIGLIP_MODEL_NAME,
  MODEL_CONFIG
);

const emptyImageInputs = {
  pixel_values: createEmptyTensor("float32", [1, 3, 224, 224]),
};

export default async function generateTextEmbedding(
  text: string
): Promise<number[]> {
  const textInputs = await textProcessor(text);
  const { text_embeds } = await model({ ...textInputs, ...emptyImageInputs });

  return text_embeds.normalize().tolist()[0];
}
