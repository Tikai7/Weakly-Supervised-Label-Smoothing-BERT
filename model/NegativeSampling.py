import re
import pandas as pd
import pyterrier as pt
import numpy as np 
    
class RandomSampling():
    @staticmethod
    def sample(data, k=2):
        data['random_score'] = np.random.uniform(0, 1, size=len(data))
        duplicates = data[data['is_duplicate'] == 1]
        new_question1 = []
        new_question2 = []
        new_labels = []
        random_scores = []
        for _, row in duplicates.iterrows():
            sampled_rows = duplicates.sample(n=k, replace=False)  # replace=True to allow choosing the same row more than once if needed
            for _, other_row in sampled_rows.iterrows():
                if row['question1'] != other_row['question1']:  # Ensure not to duplicate the exact pair
                    new_question1.append(row['question1'])
                    new_question2.append(other_row['question2'])
                    new_labels.append(0)  # These are non-duplicate by design
                    random_scores.append(np.random.uniform(0, 1))
        augmented_data = pd.DataFrame({
            'question1': new_question1,
            'question2': new_question2,
            'is_duplicate': new_labels,
            'score': random_scores
        })
        final_df = pd.concat([data, augmented_data], ignore_index=True)
        # Drop the 'global_docno' column before returning
        final_df = final_df.drop(columns="random_score")
        return final_df

class BM25Sampling():
    @staticmethod
    def preprocess_query(query):
        # Basic preprocessing to ensure consistent query formatting
        query = query.lower()
        query = re.sub(r'[^\w\s]', '', query)  # remove punctuation
        return query
    
    @staticmethod
    def sample(index_ref, data, k=2):
        index = pt.IndexFactory.of(index_ref)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=['docno'])
        
        new_questions = []

        # Iterate through each row in the dataset
        for _, row in data.iterrows():
            # Prepare the query from question1
            query_df = pd.DataFrame({'query': [BM25Sampling.preprocess_query(row['question1'])], 'qid': [1]})
            results = bm25.transform(query_df)
            
            # Filter results to get top k valid results that are not self-references
            valid_results = results[results['docno'].isin(data['global_docno'])].head(k)
            for _, result in valid_results.iterrows():
                if result['docno'] != row['global_docno']:
                    matched_row = data[data['global_docno'] == result['docno']].iloc[0]
                    new_questions.append({
                        'question1': row['question1'],
                        'question2': matched_row['question2'],
                        'is_duplicate': 0,
                        'score': result['score'] if pd.notna(result['score']) else np.random.uniform()
                    })

            # Also store the score for the original pair if it's a duplicate
            if row['is_duplicate'] == 1:
                self_scores = results[results['docno'] == row['global_docno']]['score'].values
                new_questions.append({
                    'question1': row['question1'],
                    'question2': row['question2'],
                    'is_duplicate': 1,
                    'bm25_score': self_scores[0] if (len(self_scores) > 0 and pd.notna(self_scores[0])) else 1
                })

        augmented_data = pd.DataFrame(new_questions)

        if 'bm25_score' in augmented_data and not augmented_data.empty:
            max_score = augmented_data['bm25_score'].max()
            min_score = augmented_data['bm25_score'].min()
            if max_score > min_score:  # Prevent division by zero
                augmented_data['score'] = (augmented_data['score'] - min_score) / (max_score - min_score)
            else:
                augmented_data['score'] = 0.0  # If all scores are the same

        # Combine with the original data
        final_df = pd.concat([data, augmented_data], ignore_index=True)
        # Drop the 'global_docno' column before returning
        final_df = final_df.drop(columns=["global_docno","bm25_score"])
        
        return final_df