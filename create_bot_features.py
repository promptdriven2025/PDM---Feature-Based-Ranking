import logging
import sys
from optparse import OptionParser
import gensim
from tqdm import tqdm
import utils
from gen_utils import run_bash_command, list_multiprocessing
import os
from copy import deepcopy
from vector_functionality import query_term_freq, centroid_similarity, calculate_similarity_to_docs_centroid_tf_idf \
    , document_centroid, calculate_semantic_similarity_to_top_docs, get_text_centroid, add_dict, cosine_similarity
from utils import clean_texts, read_trec_file, load_file, get_java_object, create_trectext, create_index, run_model, \
    create_features_file_diff, \
    read_raw_trec_file, create_trec_eval_file, order_trec_file, retrieve_scores, transform_query_text, \
    read_queries_file, create_index_to_query_dict, parse_with_regex
from nltk import sent_tokenize
import numpy as np
import math
from multiprocessing import cpu_count
from functools import partial



def custom_split(s):
    return s.rsplit('_', 2)[0], s.rsplit('_', 2)[1], s.rsplit('_', 2)[2]


def create_sentence_pairs(top_docs, ref_doc, texts):
    result = {}
    result_texts = {}
    ref_sentences = sent_tokenize(texts[ref_doc])
    for doc in top_docs:
        doc_sentences = sent_tokenize(texts[doc])
        for i, top_sentence in enumerate(doc_sentences):
            for j, ref_sentence in enumerate(ref_sentences):
                # key = ref_doc + "$" + doc + "_" + str(j) + "_" + str(i)
                key = ref_doc + "$" + doc + "_" + str(i + 1) + "_" + str(j + 1)  # fix to fit qrels

                result[key] = ref_sentence.rstrip().replace("\n", "") + "\t" + top_sentence.rstrip().replace("\n", "")
                text_lst = ref_sentences.copy()
                text_lst[j] = top_sentence
                result_texts[key] = ' '.join(text_lst).rstrip().replace("\n", "")
    return result, result_texts




def create_raw_dataset(ranked_lists, doc_texts, output_file="raw_ds_out.txt", ref_index=-1, top_docs_index=3,
                       is_train=True):
    ref_to_ind = {-1: 4, -2: 3, -3: 2, 1: 1}

    with open(output_file, 'w') as output:
        with open(output_file.replace(".txt", "_texts.txt"), 'w') as output_texts:
            for epoch in ranked_lists:
                if is_train:
                    rel_eps = ["06"]
                else:
                    rel_eps = ["04"]  
                if epoch not in rel_eps:
                    continue
                for query in ranked_lists[epoch]:
                    top_docs = ranked_lists[epoch][query][:top_docs_index]
                    ref_doc = ranked_lists[epoch][query][ref_index]
                    pairs, pairs_text = create_sentence_pairs(top_docs, ref_doc, doc_texts)
                    for key in pairs:
                        output.write(str(int(query)) + str(epoch) + "\t" + key + "\t" + pairs[key] + "\n")
                        output_texts.write(str(int(query)) + str(epoch) + "\t" + key + "\t" + pairs_text[key] + "\n")



def read_raw_ds(raw_dataset):
    result = {}
    with open(raw_dataset, encoding="utf-8") as ds:
        for line in ds:
            key = line.split("\t")[1]
            query = key.split("-")[2]
            epoch = key.split("-")[1]
            comp_id = query + "_" + epoch
            if comp_id not in result:
                result[comp_id] = {}
            sentence_out = line.split("\t")[2]
            sentence_in = line.split("\t")[3].rstrip()
            result[comp_id][key] = {"in": sentence_in, "out": sentence_out}
    return result


def reverese_query(qid):
    epoch = str(qid)[-2:]
    query = str(qid)[:-2].zfill(3)
    return epoch, query


def context_similarity(replacement_index, ref_sentences, sentence_compared, mode, model, stemmer=None):
    try:
        if mode == "own":
            ref_sentence = ref_sentences[replacement_index]
            return centroid_similarity(clean_texts(ref_sentence), clean_texts(sentence_compared), model, stemmer)
        if mode == "pred":
            if replacement_index + 1 == len(ref_sentences):
                sentence = ref_sentences[replacement_index]
            else:
                sentence = ref_sentences[replacement_index + 1]
            return centroid_similarity(clean_texts(sentence), clean_texts(sentence_compared), model, stemmer)
        if mode == "prev":
            if replacement_index == 0:
                sentence = ref_sentences[replacement_index]
            else:
                sentence = ref_sentences[replacement_index - 1]
            return centroid_similarity(clean_texts(sentence), clean_texts(sentence_compared), model, stemmer)
    except:
        print("ERROR", "replacement_index:", replacement_index, "len ref_sentences:", len(ref_sentences),"mode:", mode)


def get_past_winners(ranked_lists, epoch, query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = "0" + str(iteration + 1)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners


def create_weighted_dict(dict, weight):
    result = {}
    for token in dict:
        result[token] = float(dict[token]) * weight
    return result


def get_past_winners_tfidf_centroid(past_winners, docuemnt_vectors_dir):
    result = {}
    decay_factors = [0.01 * math.exp(-0.01 * (len(past_winners) - i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i, doc in enumerate(past_winners):
        doc_tfidf = get_java_object(docuemnt_vectors_dir + doc)
        decay = decay_factors[i] / denominator
        normalized_vector = create_weighted_dict(doc_tfidf, decay)
        result = add_dict(result, normalized_vector)
    return result


def past_winners_centroid(past_winners, texts, model, stemmer=None):
    sum_vector = None
    decay_factors = [0.01 * math.exp(-0.01 * (len(past_winners) - i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i, doc in enumerate(past_winners):
        text = texts[doc]
        vector = get_text_centroid(clean_texts(text), model, stemmer)
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector += vector * decay_factors[i] / denominator
    return sum_vector


def write_files(feature_list, feature_vals, output_dir, qid, ref):
    epoch, query = reverese_query(qid)
    ind_name = {-1: "5", 1: "2", -3: "3", -2: "4"}

    query_write = query + str(int(epoch)) + ind_name[ref]
    for feature in feature_list:
        with open(output_dir + "/doc_" + feature + "_" + query_write, 'w') as out:
            for pair in feature_vals[feature]:
                out.write(pair + " " + str(feature_vals[feature][pair]) + "\n")


def create_features(raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir,
                    tfidf_sentence_dir, queries, output_dir, qid):
    global word_embd_model
    feature_vals = {}
    relevant_pairs = raw_ds[qid]
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]

    for feature in feature_list:
        feature_vals[feature] = {}

    qid_original, epoch = qid.split("_")
    if epoch in ["06", "07"]:
        x = -1
    if epoch in ["01"]:
        return
    past_winners = get_past_winners(ranked_lists, epoch, qid_original)
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners, doc_texts, word_embd_model, True)
    past_winners_tfidf_centroid_vector = get_past_winners_tfidf_centroid(past_winners, doc_tfidf_vectors_dir)
    top_docs = ranked_lists[epoch][qid_original][:top_doc_index]
    ref_doc = ranked_lists[epoch][qid_original][ref_doc_index]
    ref_sentences = sent_tokenize(doc_texts[ref_doc])
    top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir + doc) for doc in top_docs])
    for pair in relevant_pairs:
        sentence_in = relevant_pairs[pair]["in"]
        sentence_out = relevant_pairs[pair]["out"]
        in_vec = get_text_centroid(clean_texts(sentence_in), word_embd_model, True)
        out_vec = get_text_centroid(clean_texts(sentence_out), word_embd_model, True)
        replace_index = int(pair.split("_")[-1]) - 1  # fix to fit qrels
        query = queries[qid_original]

        feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", clean_texts(sentence_in),
                                                                       clean_texts(query))
        feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", clean_texts(sentence_out),
                                                                        clean_texts(query))
        feature_vals['CosineToCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2], top_docs_tfidf_centroid)
        feature_vals['CosineToCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[0] + "_" + pair.split("_")[1], top_docs_tfidf_centroid)
        feature_vals["CosineToCentroidInVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_in, top_docs,
                                                                                                doc_texts,
                                                                                                word_embd_model, True)
        feature_vals["CosineToCentroidOutVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_out, top_docs,
                                                                                                 doc_texts,
                                                                                                 word_embd_model, True)
        feature_vals['CosineToWinnerCentroidInVec'][pair] = cosine_similarity(in_vec,
                                                                              past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidOutVec'][pair] = cosine_similarity(out_vec,
                                                                               past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
            past_winners_tfidf_centroid_vector)
        feature_vals['CosineToWinnerCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[0] + "_" + pair.split("_")[1], past_winners_tfidf_centroid_vector)
        feature_vals['SimilarityToPrev'][pair] = context_similarity(replace_index, ref_sentences, sentence_in, "prev",
                                                                    word_embd_model, True)
        feature_vals['SimilarityToRefSentence'][pair] = context_similarity(replace_index, ref_sentences, sentence_in,
                                                                           "own", word_embd_model, True)
        feature_vals['SimilarityToPred'][pair] = context_similarity(replace_index, ref_sentences, sentence_in, "pred",
                                                                    word_embd_model, True)
        feature_vals['SimilarityToPrevRef'][pair] = context_similarity(replace_index, ref_sentences, sentence_out,
                                                                       "prev", word_embd_model, True)

        feature_vals['SimilarityToPredRef'][pair] = context_similarity(replace_index, ref_sentences, sentence_out,
                                                                       "pred", word_embd_model, True)
    write_files(feature_list, feature_vals, output_dir, qid, ref_doc_index)


def feature_creation_parallel(raw_dataset_file, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
                              doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir,
                              output_final_features_dir, workingset_file):
    global word_embd_model
    args = [qid for qid in queries]
    if not os.path.exists(output_feature_files_dir):
        os.makedirs(output_feature_files_dir)
    if not os.path.exists(output_final_features_dir):
        os.makedirs(output_final_features_dir)
    raw_ds = read_raw_ds(raw_dataset_file)

    new_args = []
    for qid in queries.keys():
        for epoch in ranked_lists.keys():
            if qid + "_" + epoch in raw_ds.keys():
                new_args.append(qid + "_" + epoch)

    create_ws(raw_ds, workingset_file, ref_doc_index)
    func = partial(create_features, raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
                   doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir)
    workers = cpu_count() - 1
    list_multiprocessing(new_args, func, workers=workers)

    ind_name = {-1: "4", 1: "2", -2: "3"}   
    command = "perl /lv_local/home/user/content_modification_code-master/scripts/generateSentences.pl " + output_feature_files_dir + " " + workingset_file
    run_bash_command(command)
    run_bash_command(f"mv features " + output_final_features_dir)
    os.rename(output_final_features_dir + '/features',
              output_final_features_dir + f'/features_{ind_name[ref_doc_index]}')


def run_svm_rank_model(test_file, model_file, predictions_folder):
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    print("##Running command: " + command + "##")
    out = run_bash_command(command)
    print("Output of ranking command: " + str(out), flush=True)
    return predictions_file


def create_index_to_doc_name_dict(features):
    doc_name_index = {}
    index = 0
    with open(features) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
    return doc_name_index


def create_sentence_vector_files(output_dir, raw_ds_file, index_path, java_path, swig_path, home_path):
    command = home_path + java_path + "/bin/java -Djava.library.path=" + swig_path + " -cp seo_indri_utils.jar PrepareTFIDFVectorsSentences " + index_path + " " + raw_ds_file + " " + output_dir
    logger.info("##Running command: " + command + "##")
    out = run_bash_command(command)
    logger.info("Command output: " + str(out))


def create_paragraph_vector_files(output_dir, raw_ds_file, index_path, java_path, swig_path, home_path):
    raw_ds_file = "/lv_local/home/user/content_modification_code-master/g_output/raw_ds_t.txt"
    command = home_path + java_path + "/bin/java -Djava.library.path=" + swig_path + " -cp seo_indri_utils.jar PrepareTFIDFVectorsReferenceDocs " + index_path + " " + raw_ds_file + " " + output_dir
    logger.info("##Running command: " + command + "##")
    out = run_bash_command(command)
    logger.info("Command output: " + str(out))


def update_texts(doc_texts, pairs_ranked_lists, sentence_data):
    new_texts = {}
    for qid in pairs_ranked_lists:
        chosen_pair = pairs_ranked_lists[qid][0]
        ref_doc = chosen_pair.split("$")[0]
        replacement_index = int(chosen_pair.split("_")[1])
        sentence_in = sentence_data[qid][chosen_pair]["in"]
        sentences = sent_tokenize(doc_texts[ref_doc])
        sentences[replacement_index] = sentence_in
        new_text = "\n".join(sentences)
        new_texts[ref_doc] = new_text
    for doc in doc_texts:
        if doc not in new_texts:
            new_texts[doc] = doc_texts[doc]
    return new_texts



def create_ws(raw_ds, ws_fname, ref):
    ind_name = {-1: "5", 1: "2", -3: "3", -2: "4"}
    if not os.path.exists(os.path.dirname(ws_fname)):
        os.makedirs(os.path.dirname(ws_fname))
    lines = []
    for qid in sorted(raw_ds):
        epoch, query = reverese_query(qid)
        if epoch == '01':
            continue
        query_write = query + str(int(epoch)) + ind_name[ref]
        for i, pair in enumerate(sorted(raw_ds[qid])):
            out_ = str(int(pair.split("_")[1]))
            in_ = str(int(pair.split("_")[2]))
            name = pair.split("$")[1].split("_")[0] + "_" + out_ + "_" + in_  
            s = query_write + " Q0 " + name + " 0 " + str(i) + " pairs_seo\n"
            if type(s) == str:
                lines.append(s)
    with open(ws_fname, 'w') as ws:
        ws.writelines(lines)


def update_text_doc(text, new_sentence, index):
    sentences = sent_tokenize(text)
    sentences[index] = new_sentence
    return "\n".join(sentences)


def create_new_trectext(doc, texts, new_text, new_trectext_name):
    text_copy = deepcopy(texts)
    text_copy[doc] = new_text
    create_trectext(text_copy, new_trectext_name, "ws_debug")


def qid_matching(comp_id):
    qid, epoch = comp_id.split("_")
    return str(int(qid) * 100 + int(epoch))


#
def create_specifi_ws(qid, ranked_, fname):
    new_quid = qid_matching(qid)
    with open(fname, 'w') as out:
        for i, doc in enumerate(ranked_):
            out.write(new_quid + " Q0 " + doc + " 0 " + str(i + 1) + " pairs_seo\n")


def run_reranking(new_index, sentence_in, qid, specific_ws, ref_doc, out_index, texts, new_trectext_name, ranked_,
                  new_feature_file, feature_dir, trec_file, score_file, options):
    new_text = update_text_doc(texts[ref_doc], sentence_in, out_index)
    create_new_trectext(ref_doc, texts, new_text, new_trectext_name)
    create_specifi_ws(qid, ranked_, specific_ws)
    logger.info("creating features")
    create_index(new_trectext_name, os.path.dirname(new_index), os.path.basename(new_index),
                 options.home_path, options.indri_path)
    features_file = create_features_file_diff(feature_dir, options.index_path, new_index,
                                              new_feature_file, specific_ws, options.scripts_path, options.java_path,
                                              options.swig_path, options.stopwords_file, options.queries_text_file,
                                              options.home_path)
    logger.info("creating docname index")
    docname_index = create_index_to_doc_name_dict(features_file)
    query_index = create_index_to_query_dict(features_file)
    logger.info("docname index creation is completed")
    logger.info("features creation completed")
    logger.info("running ranking model on features file")
    score_file = run_model(features_file, options.home_path, options.java_path, options.jar_path, score_file,
                           options.model)
    logger.info("ranking completed")
    logger.info("retrieving scores")
    scores = retrieve_scores(docname_index, query_index, score_file)
    logger.info("scores retrieval completed")
    logger.info("creating trec_eval file")
    tmp_trec = create_trec_eval_file(scores, trec_file)
    logger.info("trec file creation is completed")
    logger.info("ordering trec file")
    final = order_trec_file(tmp_trec)
    logger.info("ranking procedure completed")
    return final
#
#
def create_qrels(raw_ds,base_trec,out_file,ref,new_indices_dir,texts,options): #TODO: changed by me!!!
    ind_name = {-1: "5", 1: "2"}
    with open(out_file,'w') as qrels:
        ranked_lists = read_raw_trec_file(base_trec)
        ranked_lists = {inner_key: inner_value[:5] if len(inner_value) >= 5 else inner_value + [None] * (5 - len(inner_value)) for
            inner_key, inner_value in ranked_lists.items()}
        raw_stats = read_raw_ds(raw_ds)
        ws_dir = "tmp_ws/"
        if not os.path.exists(ws_dir):
            os.makedirs(ws_dir)
        trectext_dir = "tmp_trectext/"
        if not os.path.exists(trectext_dir):
            os.makedirs(trectext_dir)
        trec_dir = "tmp_trec/"
        if not os.path.exists(trec_dir):
            os.makedirs(trec_dir)
        scores_dir = "tmp_scores/"
        if not os.path.exists(scores_dir):
            os.makedirs(scores_dir)
        for qid in raw_stats:
            epoch,q=reverese_query(qid)

            """ Change FOR GENERIC porpuses if needed"""
            if epoch not in ["04","06"]:
                continue

            for pair in raw_stats[qid]:
                ref_doc = pair.split("$")[0]
                out_index = int(pair.split("_")[1])
                epoch, query = reverese_query(qid)
                query_write = query + str(int(epoch)) + ind_name[ref]
                out_ = str(int(pair.split("_")[1]))
                in_ = str(int(pair.split("_")[2]))
                name = pair.split("$")[1].split("_")[0] + "_" + in_ + "_" + out_
                fname_pair = pair.replace("$","_")
                feature_dir = "tmp_features/" +fname_pair+"/"
                if not os.path.exists(feature_dir):
                    os.makedirs(feature_dir)
                features_file = "qrels_features/"+fname_pair
                final_trec = run_reranking(new_indices_dir+fname_pair,raw_stats[qid][pair]["in"],qid,ws_dir+fname_pair,ref_doc,out_index,texts,trectext_dir+fname_pair,ranked_lists,features_file,feature_dir,trec_dir+fname_pair,scores_dir+fname_pair,options)
                new_lists = read_raw_trec_file(final_trec)
                label = str(max(ranked_lists[qid.replace("_","").lstrip("0")].index(ref_doc) - new_lists[qid.replace("_","").lstrip("0")].index(ref_doc), 0))
                qrels.write(query_write+" 0 "+name+" "+label+"\n")


if __name__ == "__main__":
    is_train = False

    if is_train:
        poses = [1, -1]
    else:
        poses = [1, -2, -1]  

    queries_file = "data/queries_bot_modified_sorted_t.xml"  
    embedding_model_file = "/lv_local/home/user/content_modification_code-master/embedding_models/GoogleNews-vectors-negative300.bin"
    queries = read_queries_file(queries_file)
    queries = transform_query_text(queries)
    word_embd_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_model_file, binary=True,
                                                                      limit=700000)

    for POS in tqdm(poses):

        ind_name = {-1: "4", 1: "2", -2: "3"}  
        ref_to_ind = {-1: 3, -2: 2, 1: 1} 
        print("\n\n########################## STARTING POS: " + ind_name[POS] + " ##########################\n\n")

        program = os.path.basename(sys.argv[0])
        logger = logging.getLogger(program)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % ' '.join(sys.argv))
        parser = OptionParser()

        parser.add_option("--mode", dest="mode", default="features")
        parser.add_option("--index_path", dest="index_path", default='/lv_local/home/user/INDEX')
        parser.add_option("--top_docs_index", dest="top_docs_index", default=str(min(3, ref_to_ind[POS])))
        parser.add_option("--home_path", dest="home_path", default="./")
        parser.add_option("--jar_path", dest="jar_path", default="./scripts/RankLib.jar")
        parser.add_option("--java_path", dest="java_path", default="../opt/java/jdk1.8.0")
        parser.add_option("--swig_path", dest="swig_path", default="../indri-5.6/swig/src/java")  
        parser.add_option("--stopwords_file", dest="stopwords_file", default="data/stopwords_list")
        parser.add_option("--queries_text_file", dest="queries_text_file", default="./data/working_comp_queries.txt")
        parser.add_option("--scripts_path", dest="scripts_path", default="scripts/")
        parser.add_option("--model", dest="model", default="./rank_models/model_lambdatamart")
        parser.add_option("--indri_path", dest="indri_path", default="../indri")
        parser.add_option("--doc_tfidf_dir", dest="doc_tfidf_dir", default="./asr_tfidf_vectors_t/")  
        parser.add_option("--sentences_tfidf_dir", dest="sentences_tfidf_dir",
                          default="./g_output/sentences_tfidf_dir/")
        parser.add_option("--scores_dir", dest="scores_dir", default="./g_output/scores_dir")
        parser.add_option("--trec_file", dest="trec_file", default="./trecs/trec_file_original_sorted_t.txt") 
        parser.add_option("--output_feature_files_dir", dest="output_feature_files_dir",
                          default="./g_output/output_feature_files_dir")
        parser.add_option("--output_final_feature_file_dir", dest="output_final_feature_file_dir",
                          default="./g_output/output_final_feature_file_dir")
        parser.add_option("--trectext_file", dest="trectext_file", default="./data/documents_t.trectext")  
        parser.add_option("--workingset_file", dest="workingset_file",
                          default=f"./g_output/saved_result_files/ws_output_{ind_name[POS]}.txt")
        parser.add_option("--svm_model_file", dest="svm_model_file", default="./rank_models/harmonic_competition_model")
        parser.add_option("--raw_ds_out", dest="raw_ds_out",
                          default=f"./g_output/saved_result_files/raw_ds_out_{ind_name[POS]}.txt")
        parser.add_option("--ref_index", dest="ref_index",
                          default=f"{POS}")
        (options, args) = parser.parse_args()

        ranked_lists = read_trec_file(options.trec_file)
        doc_texts = parse_with_regex(open(options.trectext_file, 'r').read())
        mode = options.mode

        if mode == "qrels":
            create_raw_dataset(ranked_lists, doc_texts, options.raw_ds_out, int(options.ref_index),
                               int(options.top_docs_index))
            create_sentence_vector_files(options.sentences_tfidf_dir, options.raw_ds_out, options.index_path,
                                         options.java_path, options.swig_path, options.home_path)
            create_qrels(options.raw_ds_out, options.trec_file, "qrels_seo_bot" + options.ref_index + ".txt",
                         int(options.ref_index), "qrels_indices/", doc_texts, options)

        if mode == "features":

            run_bash_command('rm -r ' + options.raw_ds_out)
            if not os.path.exists(options.raw_ds_out):
                create_raw_dataset(ranked_lists, doc_texts, options.raw_ds_out, int(options.ref_index),
                                   int(options.top_docs_index), is_train)
                create_sentence_vector_files(options.sentences_tfidf_dir, options.raw_ds_out, options.index_path,
                                             options.java_path, options.swig_path, options.home_path)


            feature_creation_parallel(options.raw_ds_out, ranked_lists, doc_texts, int(options.top_docs_index),
                                      int(options.ref_index), options.doc_tfidf_dir,
                                      options.sentences_tfidf_dir, queries, options.output_feature_files_dir,
                                      options.output_final_feature_file_dir, options.workingset_file)

    utils.process_files(
        '/lv_local/home/user/content_modification_code-master/g_output/output_final_feature_file_dir',
        '/lv_local/home/user/content_modification_code-master/g_output/saved_result_files')
    if is_train:
        utils.create_train_data()
