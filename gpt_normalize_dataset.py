
import pandas as pd
from suseso_utils import SUSESO_COLUMNS
from openai import OpenAI
from joblib import Parallel, delayed
import os
import time 
from dotenv import load_dotenv
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def query_to_gpt(x, client):

    relato = f"""accidente sucedió cuando : {x[SUSESO_COLUMNS[0]]}
                descripcion del accidente  : {x[SUSESO_COLUMNS[1]]}
                lugar en que se desarrollo el accidente : {x[SUSESO_COLUMNS[2]]}
                profesion del afectado : {x[SUSESO_COLUMNS[3]]}
                zonas del cuerpo afectadas por el accidente : {x[SUSESO_COLUMNS[4]]}"""
    
    try:
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Eres una asistente que me ayuda a estructurar relatos asociados a accidentes de trabajo, tienes que estructurar la información 
                                    para que esta sea coherente, el relato se compone de cinco campos con información los cuales debes de usar para crear
                                    en un solo parrafo una descripción del suceso, teniendo en cuenta el cómo fue y que partes del curpo fueron afectadas, esto hazlo 
                                    sin añadir informacion erronea. Solo utiliza palabras en español dentro del relato."""},
            {"role": "user", "content": f"""Este es el relato: {relato}"""}
        ]
        )
        return completion.choices[0].message.content
    except:
        return ""

def procces_batch(df, key, output_folder , i):

    client = OpenAI(api_key=key)
    output_filename = os.path.join(
        output_folder, f"dataset_2022_2023_{i:05}.pkl"
    )
    dataset = df.copy()
    dataset['gpt_api_text'] = dataset.apply(lambda x : query_to_gpt(x, client), axis =1 )
    dataset.to_pickle(output_filename)
    print('batch', i, 'Done !!')

if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    dataset = pd.read_pickle(os.getenv("DATASET_SOURCE_PATH"))

    
    dataset = dataset.loc[dataset.user.isin(['isl', 'ist'])]
    dataset.reset_index(inplace=True)
    dataset.drop(['index'], axis = 1, inplace=True)

    otuput_dir = './gpt_format_dataset_v2'

    if not os.path.exists(otuput_dir):
        os.makedirs(otuput_dir)

    batch_size = 100
    tasks = []
    for i, batch_id in enumerate(range(0 , len(dataset), batch_size)) :
        tasks.append(
            delayed(procces_batch)(dataset.iloc[batch_id:batch_id + batch_size], 
                                    key,
                                    otuput_dir, 
                                    i)
        )

    Parallel(n_jobs=10, verbose=11, backend="loky")(tasks)
