import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
from transformers.cache_utils import DynamicCache
import argparse
import os
import json
from transformers import BitsAndBytesConfig
import random

def get_env():
    env_dict = {}
    with open (file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

"""Hugging Face Llama model"""
HF_TOKEN = get_env()["HF_TOKEN"]
global model_name, model, tokenizer
global rand_seed

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

# Define a simplified generate function


"""Sentence-BERT for evaluate semantic similarity"""
from sentence_transformers import SentenceTransformer, util
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight sentence-transformer

def get_bert_similarity(response, ground_truth):
    # Encode the query and text
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)

    # Compute the cosine similarity between the query and text
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)

    return cosine_score.item()

from time import time

from llama_index.core import Settings

def getOpenAIRetriever(documents: list[str], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    openai.api_key = get_env()["OPENAI_API_KEY"]        
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=get_env()["OPENAI_API_KEY"], title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"
    
    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    
    return OpenAI_retriever, t2 - t1
    

def getGeminiRetriever(documents: list[str], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    GOOGLE_API_KEY = get_env()["GOOGLE_API_KEY"]
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding( model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")
    
    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    
    return Gemini_retriever, t2 - t1
    
def getBM25Retriever(documents: list[str], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

def get_kis_dataset(filepath: str):
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()
    
    return text_list, dataset

def parse_squad_data(raw):
    dataset = { "ki_text": [], "qas": [] }
    
    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                ques = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                dataset['qas'].append({"title": data['title'], "paragraph_index": tuple((k_id, p_id)) ,"question": ques, "answers": answers})
        dataset['ki_text'].append({"id": k_id, "title": data['title'], "paragraphs": article})
    
    return dataset

def get_squad_dataset(filepath: str, max_knowledge: int = None, max_paragraph: int = None, max_questions: int = None):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)
    
    print("max_knowledge", max_knowledge, "max_paragraph", max_paragraph, "max_questions", max_questions)
    
    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = max_knowledge if max_knowledge != None and max_knowledge < len(parsed_data['ki_text']) else len(parsed_data['ki_text'])
    
    # Shuffle the Articles and Questions
    if rand_seed != None:
        random.seed(rand_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])
        k_ids = [i['id'] for i in parsed_data["ki_text"][:max_knowledge]]

        
    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data['ki_text'][:max_knowledge]:
        max_para = max_paragraph if max_paragraph != None and max_paragraph < len(article['paragraphs']) else len(article['paragraphs'])
        text_list.append(article['title'])
        text_list.append('\n'.join(article['paragraphs'][0:max_para]))
    
    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [qa['question'] for qa in parsed_data['qas'] if qa['paragraph_index'][0] in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    answers = [qa['answers'][0] for qa in parsed_data['qas'] if qa['paragraph_index'][0]  in k_ids and (max_paragraph == None or qa['paragraph_index'][1] < max_paragraph)]
    
    dataset = zip(questions, answers)
    
    return text_list, dataset


def get_hotpotqa_dataset(filepath: str, max_knowledge: int = None):
    # Open and read the JSON
    with open (filepath, "r") as file:
        data = json.load(file)
    
    if rand_seed != None:
        random.seed(rand_seed)
        random.shuffle(data)
    
    questions = [ qa['question'] for qa in data ]
    answers = [ qa['answer'] for qa in data ]
    dataset = zip(questions, answers)
    
    if max_knowledge == None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))
    
    text_list = []
    for i, qa in enumerate(data[:max_knowledge]):
        context = qa['context']
        context = [ c[0] + ": \n" + "".join(c[1]) for c in context ]
        article = "\n\n".join(context)

        text_list.append(article)
    
    return text_list, dataset
    
def rag_test(args: argparse.Namespace):
    answer_instruction = None
    if args.dataset == "kis_sample":
        datapath = "./datasets/rag_sample_qas_from_kis.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "kis":
        datapath = "./datasets/synthetic_knowledge_items.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "squad-dev":
        datapath = "./datasets/squad/dev-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
    if args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-dev":
        datapath = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction
    if args.dataset == "hotpotqa-test":
        datapath = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
        
    if answer_instruction != None:
        answer_instruction = "Answer the question with a super short answer."
    
    kvcache_path = "./data_cache/cache_knowledges.pt"
    # document indexing for the rag retriever
    documents = [Document(text=t) for t in text_list]
    
    if args.index == "gemini":
        retriever, prepare_time = getGeminiRetriever(documents, similarity_top_k=args.topk)
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents, similarity_top_k=args.topk)
    if args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents, similarity_top_k=args.topk)
        
    print(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds\n")
    
    results = {
        "retrieve_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }
    
    dataset = list(dataset) # Convert the dataset to a list
    
    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion != None else len(dataset)
    
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):    # Retrieve the knowledge from the vector database
        retrieve_t1 = time()
        nodes = retriever.retrieve(question)
        retrieve_t2 = time()
        
        knowledge = "\n---------------------\n".join([node.text for node in nodes])
        # short_knowledge = knowledge[:knowledge.find("**Step 4")]
        
        prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is below.
    ------------------------------------------------
    {knowledge}
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

        # Generate Response for the question
        generate_t1 = time() 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=300,  # Set the maximum length of the generated text
            do_sample=False,  # Ensures greedy decoding,
            temperature=None
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generate_t2 = time() 

        generated_text = generated_text[generated_text.find(question) + len(question):]
        generated_text = generated_text[generated_text.find('assistant') + len('assistant'):].lstrip()
        
        # print("R: ", knowledge)
        print("Q: ", question)
        print("A: ", generated_text)
        
        # Evaluate bert-score similarity
        similarity = get_bert_similarity(generated_text, ground_truth)
        
        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t",
            f"retrieve time: {retrieve_t2 - retrieve_t1},\t",
            f"generate time: {generate_t2 - generate_t1}"
            )
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t retrieve time: {retrieve_t2 - retrieve_t1},\t generate time: {generate_t2 - generate_t1}\n")
            
        results["prompts"].append(prompt)
        results["responses"].append(generated_text)
        results["retrieve_time"].append(retrieve_t2 - retrieve_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)
        
        with open(args.output, "a") as f:
            f.write(f"[{id}]: [Cumulative]: " 
                    + f"Semantic Similarity: {round(sum(results['similarity']) / (len(results['similarity'])) , 5)}," 
                    + f"\t retrieve time: {sum(results['retrieve_time']) / (len(results['retrieve_time'])) },"
                    + f"\t generate time: {sum(results['generate_time']) / (len(results['generate_time'])) }\n")
        
        
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_retrieve_time = sum(results["retrieve_time"]) / len(results["retrieve_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}\n")


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)

def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )
    
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--index', choices=['gemini', 'openai', 'bm25'], required=True, help='Index to use (gemini, openai, bm25)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None ,type=int, help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None ,type=int, help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None ,type=int, help='Maximum number of paragraph to use')
    parser.add_argument('--topk', required=False, default=1, type=int, help='Top K retrievals to use')
    parser.add_argument('--dataset', required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)', 
                        choices=['kis', 'kis_sample', 
                                'squad-dev', 'squad-train', 
                                'hotpotqa-dev',  'hotpotqa-train', 'hotpotqa-test'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions
    
    args = parser.parse_args()
    
    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomSeed", args.randomSeed)
    
    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed != None else None
    
    if args.quantized:
        tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )
    
    def unique_path(path, i=0):
        if os.path.exists(path):
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    rag_test(args)
