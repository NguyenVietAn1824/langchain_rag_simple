import os
import wget

file_links = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762"
    },
    {
        "title": "BERT - Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805"
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "url": "https://arxiv.org/pdf/2201.11903"
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "url": "https://arxiv.org/pdf/2006.11239"
    },
    {
        "title": "Instruction Tuning for Large Language Models - A Survey",
        "url": "https://arxiv.org/pdf/2308.10792"
    }
]

def is_exit(file_link):
    return os.path.exists(f"./{file_link['title']}.pdf")

for file_link in file_links:
    if not is_exit(file_link):
        wget.download(file_link["url"], file_link["title"])