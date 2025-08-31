import { AutoProcessor, RawImage, AutoModel } from "@huggingface/transformers";

import { JINA_V2_MODEL_NAME, MODEL_CONFIG } from "@/transformers/config";

const processor = await AutoProcessor.from_pretrained(JINA_V2_MODEL_NAME);
const model = await AutoModel.from_pretrained(JINA_V2_MODEL_NAME, MODEL_CONFIG);

export default async function generateImageEmbedding(
  path: string
): Promise<number[]> {
  const image = await RawImage.read(path);
  const imageInputs = await processor(null, image);
  const { l2norm_image_embeddings } = await model(imageInputs);

  return l2norm_image_embeddings.tolist()[0];
}
