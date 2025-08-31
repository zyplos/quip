import { AutoProcessor, AutoModel } from "@huggingface/transformers";

import { JINA_V2_MODEL_NAME, MODEL_CONFIG } from "@/transformers/config";

const processor = await AutoProcessor.from_pretrained(JINA_V2_MODEL_NAME);
const model = await AutoModel.from_pretrained(JINA_V2_MODEL_NAME, MODEL_CONFIG);

export default async function generateTextEmbedding(
  text: string
): Promise<number[]> {
  const textInputs = await processor(text);
  const { l2norm_text_embeddings } = await model(textInputs);

  return l2norm_text_embeddings.tolist()[0];
}
