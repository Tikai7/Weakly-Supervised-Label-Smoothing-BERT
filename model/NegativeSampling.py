import random
import pandas as pd
import pyterrier as pt
import numpy as np 
from sklearn.preprocessing import minmax_scale


class RandomSampling():
    @staticmethod
    def augment_data(data, k=2):
        data['random_score'] = np.random.uniform(0, 1, size=len(data))
        duplicates = data[data['is_duplicate'] == 1]
        new_question1 = []
        new_question2 = []
        new_labels = []
        random_scores = []
        for _, row in duplicates.iterrows():
            sampled_rows = duplicates.sample(n=k, replace=False)
            for _, other_row in sampled_rows.iterrows():
                if row['question1'] != other_row['question1']:  
                    new_question1.append(row['question1'])
                    new_question2.append(other_row['question2'])
                    new_labels.append(0)  
                    random_scores.append(np.random.uniform(0, 1))
        augmented_data = pd.DataFrame({
            'question1': new_question1,
            'question2': new_question2,
            'is_duplicate': new_labels,
            'random_score': random_scores
        })

        return pd.concat([data, augmented_data], ignore_index=True)
    
class Sampler():
    def __init__(self,candidats,k,type_ns, type_q):
        """
        candidats: Liste des candidates parmi lesquels il faudra sample
        k: nombre de candidats à sample
        type_ns: type de negative sampling
        type_q: type de transformation de query
        """
        self.candidats = candidats
        #récupèrer k candidats parmi la liste des candidats
        self.k = k
        self.type_ns = type_ns
        self.type_q = type_q

    def create_index(self,path_index,indexref):
        """
        Creates an index for the collection with PyTerrier, to use when sampling with BM-25.
        Input:
            - path_index: chemin où placer l'index
        """
        docs = pd.DataFrame(self.candidats)
        indexer = pt.DFIndexer(f"{path_index}/index_projet", overwrite=True)         # Définition du format de données (DFIndexer())
        index_ref = indexer.index(docs["docid"], docs["body"])
        return pt.IndexFactory.of(indexref), indexref
        
    def sampling(self, query, pertinent_docs, path_index):
        """
        Input:
            query: won't be used for the random sampling but for BM-25.
            pertinent_docs: documents pertinents, si les documents sampled se trouve dans la liste des documents pertinents alors il faut en tirer de nouveaux.
        Output:
            final_sampled: liste des docs non pertinent sampled de la base (random ou alors BM-25)
            scores: scores des documents de final_sampled (score aléatoire uniforme)
            pertinent_present: booléen,True si le document pertinent a été trouvé dans la liste final_sampled, False sinon
            pertinent_doc_rank: rang du document pertinent si trouvé et son rang dans la liste final_sammpled
            
        """
        pertinent_present = False
        pertinent_doc_rank = -1
        final_sampled = []
        if str.lower(self.type_ns) == 'random':
            sampled = random.sample(self.candidats, self.k)
            #Supprimer les documents pertinent de la liste sampled.
            for i,d in enumerate(sampled):
                if d in pertinent_docs:
                    pertinent_present = True
                    pertinent_doc_rank = i
                else:
                    final_sampled.append(d)
            # Si la taille de la liste finale n'est pas égale à k (dans ce cas il y avait un document pertinent dans cette liste) alors resample encore.
            while len(final_sampled) != self.k:
                final_sampled = [d for d in random.sample(self.candidats, self.k) if d not in pertinent_docs]
            scores = [random.uniform(0,0.99) for i in range(len(final_sampled))]

        elif str.lower(self.type_ns) == 'bm25':
            if not pt.started():
                pt.init()
            #Create the index:
            index,indexref = self.create_index(path_index)
            #Apply BM_25 on the query
            if str.lower(self.type_q) == "rm3":
                pipeline = (pt.BatchRetrieve(indexref, wmodel="BM25") >> 
                    pt.rewrite.RM3(indexref) >> 
                    pt.BatchRetrieve(indexref, wmodel="BM25")
                )
            elif str.lower(self.type_q) == "bo1":
                pipeline = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "Bo1", "qe" : "on"})

            elif str.lower(self.type_q) == "kl":
                pipeline = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "KL", "qe" : "on"})
                
            elif str.lower(self.type_q) == "none":
                pipeline = pt.BatchRetrieve(index, wmodel="BM25")
            
            #Retrieve top-k docs
            sampled_initial = pipeline.search(query).head(self.k)

            #Check if a pertinent doc is present in the sampled list, and delete it from the final list
            scores = []
            for i, row in enumerate(sampled_initial.itertuples()):
                doc, score = row.docno, row.score
                if doc in pertinent_docs:  # Assuming relevant_docs is defined
                    pertinent_present = True
                    pertinent_doc_rank = i
                else:
                    final_sampled.append(doc)
                    scores.append(score)
        
            # Si la taille de la liste finale n'est pas égale à k (dans ce cas il y avait un document pertinent dans cette liste) alors resample encore.
            scores_for_random = [0] * (self.k - len(sampled))
            while len(final_sampled) != self.k:
                extra_samples = [d for d in random.sample(self.candidats, self.k - len(final_sampled))
                                 if d not in final_sampled and d not in pertinent_docs]
                final_sampled.extend(extra_samples)
        
            # Normalize scores if not empty
            if scores:
                normalized_scores = minmax_scale(scores, feature_range=(0.01, 0.99))
            else:
                normalized_scores = []
            normalized_scores.extend(scores_for_random)
            scores = normalized_scores
            
        return final_sampled, pertinent_present, pertinent_doc_rank, scores