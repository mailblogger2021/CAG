# Cache-Augmented Generation (CAG)

Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for enhancing language models by integrating external knowledge sources. However, RAG also introduces several challenges, including:  
- **Retrieval Latency** – Delays caused by real-time retrieval steps.  
- **Retrieval Errors** – Inaccuracies in selecting relevant documents.  
- **System Complexity** – Increased architectural and maintenance overhead.  

To address these limitations, we propose **Cache-Augmented Generation (CAG)**—an alternative paradigm that bypasses real-time retrieval. CAG leverages the extended context windows of modern large language models (LLMs) by preloading all relevant resources into the model’s context and caching its runtime parameters. During inference, the preloaded KV-cache enables the model to generate responses directly, eliminating the need for retrieval.  

**Advantages of CAG**  
- **Reduced Latency** – Eliminates real-time retrieval, enabling faster inference.  
- **Improved Reliability** – Minimizes retrieval errors while maintaining context relevance.  
- **Simplified Design** – Provides a streamlined, retrieval-free alternative to RAG, achieving comparable or superior results with lower complexity.  

**Limitations of CAG**  
- **Limited Knowledge Size** – CAG requires the entire knowledge source to fit within the context window, making it less suitable for tasks involving extremely large datasets.  
- **Context Length Constraints** – The performance of LLMs may degrade with very long contexts ([reference](https://arxiv.org/pdf/2404.02060v2)).  

Our [paper](https://arxiv.org/abs/2412.15605) investigates the relationship between model performance and context length, providing insights into scenarios where CAG excels.  

The limitations of CAG are rapidly being addressed by advancements in LLMs with longer context windows and improved capabilities for extracting relevant information from extended inputs. As these models continue to evolve, **CAG** is expected to handle increasingly complex applications, making it a practical and scalable alternative to traditional RAG.  

---

## Installation 
```bash
pip install -r ./requirements.txt
```

## Preparation
> [!IMPORTANT]  
> download the required `squad` and `hotpotqa` datasets by curl script
> ```bash
> sh ./downloads.sh
> ```

> [!IMPORTANT]
> create `.env` file by `.env.template` and input the keys required
> ```bash
> cp ./.env.template ./.env
> ```

## Usage
- `rag.py` is for RAG Experiment
- `kvcache.py` is for CAG Experiment

## Parameter Usage -- kvcache.py
- `--kvcache`: "file"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string
- `--usePrompt`, add this parameter if not using CAG knowledge cache acceleration 

### Example -- kvcache.py
```bash
python ./kvcache.py --kvcache file --dataset "squad-train" --similarity bertscore \
    --maxKnowledge 5 --maxParagraph 100 --maxQuestion 1000  \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed 0 \
    --output "./result_kvcache.txt"
```

## Parameter Usage -- rag.py
- `--index`: "openai" or "bm25"
- `--dataset`: "hotpotqa-train" or "squad-train"
- `--similarity` "bertscore"
- `--maxKnowledge`: "", int, select how many document in dataset, explanation in Note
- `--maxParagraph`: 100
- `--maxQuestion` int, max question number, explanation in Note
- `--topk`: int, the similarity topk of retrieval
- `--modelname`: "meta-llama/Llama-3.1-8B-Instruct"
- `--randomSeed`: "", int, a random seed number
- `--output`: "", str, output filepath string

### Example -- rag.py
```bash
python ./rag.py --index "bm25" --dataset "hotpotqa-train" --similarity bertscore \
    --maxKnowledge 80 --maxParagraph 100 --maxQuestion 80 --topk 3 \
    --modelname "meta-llama/Llama-3.1-8B-Instruct" --randomSeed  0 \
    --output  "./rag_results.txt"
```

### Note:
#### `--maxKnowledge` parameter notice: 
> [!NOTE]
> Approximate Tokens count corresponding to knowledge document size of "squad-train" and "hotpotqa-train" dataset. 

> datasets=("squad-train")
> - when k = 3, tokens = 21,000
> - when k = 4, tokens = 32,000
> - when k = 7, tokens = 50,000
> 
> datasets=("hotpotqa-train")
> - all k = 7405 article, tokens = 10,038,084 
> - when k = 1, tokens = 1,400
> - when k = 16, tokens = 22,400
> - when k = 24, tokens = 33,667
> - when k = 32, tokens = 44,800
> - when k = 48, tokens = 64,000
> - when k = 64, tokens = 85,000
> - when k = 80, tokens = 106,000

#### `--maxQuestion` parameter notice:
> - when using "squad-train" dataset, 1 knowledge has average 150 questions
> - when using "hotpotqa-train" dataset, 1 knowledge has 1 question

> [!TIP]
> Since 1 document in "hotpoqa-train" dataset has only 1 question, it may not satisfy large-scale evaluation.
> Multiple evaluation could be a relatively better approach.
> 

## Citation
```
@misc{chan2024dontragcacheaugmentedgeneration,
      title={Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks}, 
      author={Brian J Chan and Chao-Ting Chen and Jui-Hung Cheng and Hen-Hsen Huang},
      year={2024},
      eprint={2412.15605},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.15605}, 
}
```


