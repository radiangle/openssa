from argparse import ArgumentParser
from functools import cache

# pylint: disable=wrong-import-order
from data import DocName, FbId, Answer, FB_ID_COL_NAME, DOC_NAMES_BY_FB_ID, QS_BY_FB_ID, LOCAL_CACHE_DOCS_DIR_PATH,  cache_dir_path
from util import enable_batch_qa, log_qa_and_update_output_file
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


def llama_answer(question: str, fb_id: int, n_words: int = 300) -> str:  # noqa: ARG001
        """Answer question by using generic llamaindex from file-stored informational resource."""
        doc_name = DOC_NAMES_BY_FB_ID[fb_id]
        doc_path: Path = LOCAL_CACHE_DOCS_DIR_PATH / doc_name  # noqa: F821

        documents = SimpleDirectoryReader(doc_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        return query_engine.query(question).response


@enable_batch_qa
@log_qa_and_update_output_file(output_name='llamaindex-generic')
def answer(fb_id: FbId) -> Answer:
    return (llama_answer(QS_BY_FB_ID[fb_id], fb_id))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('fb_id')
    args = arg_parser.parse_args()

    answer(fb_id
           if (fb_id := args.fb_id).startswith(FB_ID_COL_NAME)
           else f'{FB_ID_COL_NAME}_{fb_id}')
