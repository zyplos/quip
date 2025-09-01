// import siglipDifferentTextEmbedder from "@/transformers/siglip-different-models/generateTextEmbedding";
// import siglipDifferentImageEmbedder from "@/transformers/siglip-different-models/generateImageEmbedding";
// //
// import clipDifferentTextEmbedder from "@/transformers/clip-different-models/generateTextEmbedding";
// import clipDifferentImageEmbedder from "@/transformers/clip-different-models/generateImageEmbedding";
// //
// import clipProjectedTextEmbedder from "@/transformers/clip-models-with-projection/generateTextEmbedding";
// import clipProjectedImageEmbedder from "@/transformers/clip-models-with-projection/generateImageEmbedding";
// //
// import siglipModelTextEmbedder from "@/transformers/siglip-model/generateTextEmbedding";
// import siglipModelImageEmbedder from "@/transformers/siglip-model/generateImageEmbedding";
// //
// import jinaV2AutoTextEmbedder from "@/transformers/jina-v2-auto/generateTextEmbedding";
// import jinaV2AutoImageEmbedder from "@/transformers/jina-v2-auto/generateImageEmbedding";
//
import jinaV1CLIPTextEmbedder from "@/transformers/jina-v1-clip/generateTextEmbedding";
import jinaV1CLIPImageEmbedder from "@/transformers/jina-v1-clip/generateImageEmbedding";

export const generateTextEmbedding = jinaV1CLIPTextEmbedder;
export const generateImageEmbedding = jinaV1CLIPImageEmbedder;
