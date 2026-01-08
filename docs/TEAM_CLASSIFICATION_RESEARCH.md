| Feature | CLIP | Florence-2 | Qwen2-VL | SipLIP |
|---------|------|------------|----------|--------|
| Model Approach | Cluster embeddings of players; good at zero-shot classification | Prompt-based; can map images and text to outputs | Prompt-based; multimodal; works for images and videos | Cluster embeddings of players like CLIP; Has sigmoid contrastive loss, which scales better for large batch sizes |
| Inference speed | Very fast | Medium | Slow | Very fast |
| Accuracy | Very good for visual similarity but not very good at OCR | Very good for logos and reading text | Greatest accuracy | Very good for visual similarity |
| Memory | Low | Medium | High | Low |