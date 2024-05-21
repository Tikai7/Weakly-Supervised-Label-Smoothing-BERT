import re
import pandas as pd
import pyterrier as pt
import numpy as np 
    
class RandomSampling():
    @staticmethod
    def sample(data, k=2):
        if 'is_duplicate' not in data.columns:
            raise ValueError("Input dataframe must contain 'is_duplicate' column.")
        if k <= 0:
            raise ValueError("Parameter k must be a positive integer.")

        # Ajout d'une colonne score avec des valeurs aléatoires.
        data['score'] = np.random.uniform(0, 1, size=len(data))
        
        # garder que les query avec is_duplicate = 1
        duplicates = data[data['is_duplicate'] == 1]

        # Initialisation des listes pour stocker les nouvelles paires
        new_question1 = []
        new_question2 = []
        new_labels = []
        random_scores = []

        # Choisir au maximum k paires pour éviter des boucles inefficaces.
        if len(duplicates) < k:
            k = len(duplicates)

        # Utilisation de sample sans remplacement pour toute la liste en une seule fois.
        sampled_indices = np.random.choice(duplicates.index, size=k, replace=False)
        sampled_duplicates = duplicates.loc[sampled_indices]

        # Comparer chaque élément avec tous les autres, exclure la comparaison avec lui-même.
        for i, row_i in sampled_duplicates.iterrows():
            for j, row_j in sampled_duplicates.iterrows():
                if i != j:
                    new_question1.append(row_i['question1'])
                    new_question2.append(row_j['question2'])
                    new_labels.append(0)  # Ces paires sont des non-duplicatas.
                    random_scores.append(np.random.uniform(0, 1))

        # Création du nouveau DataFrame avec les paires augmentées.
        augmented_data = pd.DataFrame({
            'question1': new_question1,
            'question2': new_question2,
            'is_duplicate': new_labels,
            'score': random_scores
        })

        final_df = pd.concat([data, augmented_data], ignore_index=True)
        final_df = final_df.drop(columns="global_docno", errors="ignore")
        return final_df

class BM25Sampling():
    @staticmethod
    def preprocess_query(query):
        # Préprocess simple
        query = query.lower()
        query = re.sub(r'[^\w\s]', '', query) 
        return query
    
    @staticmethod
    def sample(index_ref, data, k=2):
        index = pt.IndexFactory.of(index_ref)
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=['docno'])
        
        new_questions = []
        # Sur les query avec is_duplicate = 1
        duplicates = data[data['is_duplicate'] == 1]
        for _, row in duplicates.iterrows():
            # faire une requete pour avoir les top k des questions 2 similaires
            query_df = pd.DataFrame({'query': [BM25Sampling.preprocess_query(row['question1'])], 'qid': [1]})
            results = bm25.transform(query_df)
            # verifié que les questions ne sont pas les mêmes + que le docno est dans global_docno (indice global avant le split du dataset)
            valid_results = results[results['docno'].isin(data['global_docno']) & (results['docno'] != row['global_docno'])].head(k)
            for _, result in valid_results.iterrows():
                matched_row = data[data['global_docno'] == result['docno']].iloc[0]
                new_questions.append({
                    'question1': row['question1'],
                    'question2': matched_row['question2'],
                    'is_duplicate': 0,
                    'score': result['score'] if pd.notna(result['score']) else np.random.uniform()
                })
        augmented_data = pd.DataFrame(new_questions)
        final_df = pd.concat([data, augmented_data], ignore_index=True)

        # normaliser les scores entre 0 et 1 
        if 'score' in final_df:
            final_df['score'].fillna(final_df.apply(lambda x: np.random.uniform() if x['is_duplicate'] == 0 else 1, axis=1), inplace=True)
            max_score = final_df['score'].max()
            min_score = final_df['score'].min()
            if max_score > min_score:  
                final_df['score'] = (final_df['score'] - min_score) / (max_score - min_score)
            else:
                final_df['score'] = 0.0  

        final_df = final_df.drop(columns="global_docno")
        return final_df