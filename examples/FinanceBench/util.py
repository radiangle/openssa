from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from pandas import DataFrame, read_csv
from tqdm import tqdm

from data import FbId, Answer, FB_ID_COL_NAME, META_DF, FB_IDS, DOC_NAMES_BY_FB_ID, QS_BY_FB_ID, OUTPUT_FILE_PATH, LOCAL_CACHE_DIR_PATH
from pathlib import Path

type QAFunc = Callable[[FbId], Answer]


def enable_batch_qa(qa_func: QAFunc) -> QAFunc:
    def decorated_qa_func(fb_id: FbId) -> Answer:
        if 'all' in fb_id.lower():
            for _fb_id in tqdm(FB_IDS):
                qa_func(_fb_id)

            return None

        return qa_func(fb_id)

    return decorated_qa_func


@dataclass
class log_qa_and_update_output_file:  # noqa: N801
    # pylint: disable=invalid-name
    output_name: str

    def __call__(self, qa_func: QAFunc) -> QAFunc:
        def decorated_qa_func(fb_id: FbId) -> Answer:
            logger.info(f'\n{DOC_NAMES_BY_FB_ID[fb_id]}:\n{QS_BY_FB_ID[fb_id]}\n'
                        f'\n{self.output_name.upper()}:\n{(answer := qa_func(fb_id))}\n')
            OUTPUT_FILE_PATH_EXP =  LOCAL_CACHE_DIR_PATH / Path (self.output_name + '_output.csv')  # noqa: N806, F823
            if OUTPUT_FILE_PATH_EXP.is_file():
                output_df: DataFrame = read_csv(OUTPUT_FILE_PATH_EXP, index_col=FB_ID_COL_NAME)

            else:
                output_df: DataFrame = META_DF[['doc_name', 'question', 'evidence_text', 'page_number', 'answer']]
                output_df.loc[:, self.output_name] = None

            output_df.loc[fb_id, self.output_name] = answer

            output_df.to_csv(OUTPUT_FILE_PATH_EXP, index=True)

            return answer

        return decorated_qa_func
