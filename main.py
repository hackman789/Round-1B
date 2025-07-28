import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


def parse_arguments():

    parser = argparse.ArgumentParser(description="Persona-Driven Document Intelligence System")
    parser.add_argument("--docs_folder", type=str, required=True, help="Path to the folder containing PDF documents.")
    parser.add_argument("--persona", type=str, required=True, help="Description of the user persona.")
    parser.add_argument("--job", type=str, required=True, help="The job-to-be-done for the persona.")
    parser.add_argument("--output_file", type=str, default="output.json", help="Path to save the output JSON file.")
    return parser.parse_args()


def extract_and_chunk_pdfs(docs_folder):

    document_chunks = []
    pdf_files = list(Path(docs_folder).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the directory: {docs_folder}")

    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:

                paragraphs = text.split('\n\n')
                for para_text in paragraphs:
                    if len(para_text.strip()) > 50:  # Filter out very short/empty paragraphs
                        document_chunks.append({
                            "doc_name": pdf_path.name,
                            "page_num": page_num + 1,
                            "content": para_text.strip()
                        })
    return document_chunks, [p.name for p in pdf_files]


def rank_chunks(chunks, persona, job):

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


    query = f"User Persona: {persona}. Specific Task: {job}"


    corpus = [chunk['content'] for chunk in chunks]


    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)


    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]


    ranked_results = []
    for hit in search_hits:
        chunk_index = hit['corpus_id']
        ranked_results.append({
            "score": hit['score'],
            "chunk_info": chunks[chunk_index]
        })
    return ranked_results


def format_output(ranked_results, input_docs, persona, job, output_file):

    extracted_sections = []
    for i, result in enumerate(ranked_results):
        chunk = result["chunk_info"]
        section_title = ' '.join(chunk["content"].split()[:8]) + '...'
        extracted_sections.append({
            "document": chunk["doc_name"],
            "page_number": chunk["page_num"],
            "section_title": section_title,
            "importance_rank": i + 1,
            "relevance_score": round(result["score"], 4)
        })


    sub_section_analysis = []
    for result in ranked_results[:5]:  # Top 5 for detailed analysis
        chunk = result["chunk_info"]
        sub_section_analysis.append({
            "document": chunk["doc_name"],
            "page_number": chunk["page_num"],
            "refined_text": chunk["content"]
        })

    output_data = {
        "metadata": {
            "input_documents": input_docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "sub_section_analysis": sub_section_analysis
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Analysis complete. Output saved to {output_file}")


if __name__ == "__main__":
    args = parse_arguments()

    print("1. Starting document analysis...")
    all_chunks, doc_names = extract_and_chunk_pdfs(args.docs_folder)
    print(f"   - Found and processed {len(doc_names)} documents.")

    print("2. Ranking content with AI model...")
    ranked = rank_chunks(all_chunks, args.persona, args.job)
    print("   - Semantic ranking complete.")

    print("3. Formatting and saving output...")
    format_output(ranked, doc_names, args.persona, args.job, args.output_file)