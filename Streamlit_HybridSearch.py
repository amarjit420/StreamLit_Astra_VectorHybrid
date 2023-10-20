import os
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import streamlit as st
from streamlit_chat import message
from itertools import cycle
import pandas as pd

session = None
cluster = None
text_emb = None
lst = None
img_model = SentenceTransformer('clip-ViT-B-32')

PATH_TO_BUNDLE      = "<location to>/secure-connect-cassio-v1.zip"
ASTRA_CLIENT_ID     = '<Astra ID>'
ASTRA_CLIENT_SECRET = '<Astra Secret>'
KEYSPACE_NAME       = 'ks3'
TABLE_NAME          = 'hybridsearch'
INPUT_PATH          = "<image location path>"


def connect_astra():
    global session
    global cluster

    # Replace these values with the path to your secure connect bundle and the database credentials
    SECURE_CONNECT_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), PATH_TO_BUNDLE)

    # Connect to the database
    cloud_config = {'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH}
    auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()

    print(f"Creating table {TABLE_NAME} in keyspace {KEYSPACE_NAME}")


def shutdown():
    global session
    global cluster

    # Close the connection
    session.shutdown()
    cluster.shutdown()

def insert_data():
    global lst
    lst = []

    # Iterate over each file in the directory
    for filename in os.listdir(INPUT_PATH):
        # Check if the file is a .jpg image
        if filename.endswith('.jpg'):
            image = Image.open(INPUT_PATH + filename)

            # Extract file name without extension
            file_name_without_extension = os.path.splitext(filename)[0]

            print(f"Processing image: {file_name_without_extension}")

            doc = {}
            embedding = image_embedding(image, img_model)
            formatted_string   = file_name_without_extension.split("_")
            doc['colour']      = formatted_string[0]
            doc['description'] = ' '.join(formatted_string[1:])
            doc['embedding']   = embedding.tolist()

            lst.append(doc)
            print(doc)

    count = 0

    for data in lst:
        count += 1
        print(f"Data #{count}:")
        image_colour = data["colour"]
        image_description = data["description"]
        image_embeddings = data["embedding"]

        # Perform operations with the data
        print("Image Colour:", image_colour)
        print("Image Desc:", image_description)
        print("Image Embedding (First 20 characters):", str(image_embeddings)[:50])
        print()

        # Insert the data into the table
        session.execute(
            f"INSERT INTO {KEYSPACE_NAME}.{TABLE_NAME} (id, colour, description, item_vector) VALUES (%s, %s, %s, %s)",
            (count, image_colour, image_description, image_embeddings))

    print("Total number of records inserted :", count)

def ann_response(input):
    vector_query_str = "round cake that's red"
    print("Query String: ", vector_query_str)
    text_emb = img_model.encode(vector_query_str)
    print(f"ANN model provided embeddings for the string: 'search': {text_emb.tolist()}")


    # Retrieve the nearest matching image from the database
    query = f"SELECT colour, description FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {text_emb.tolist()} LIMIT 3"
    print(query)

    result = session.execute(query)

    for row in result:
        print("Colour:", row.colour)
        print("Desc:", row.description)

def text_response(input):
    global session
    print("Analyzer Match query String: round and edible")

    query = f"SELECT colour, description FROM {KEYSPACE_NAME}.{TABLE_NAME} WHERE description : 'round' AND description : 'edible'  LIMIT 3"
    print(query)
    result = session.execute(query)

    if result is not None:
        for row in result:
            print("Colour:", row.colour)
            print("Desc:", row.description)

def hybrid_response(text):
    print("Analyzer Match and ANN together ")

    query = f"SELECT colour, description FROM {KEYSPACE_NAME}.{TABLE_NAME} WHERE description : 'round' AND description : 'edible' AND colour = 'red'  ORDER BY item_vector ANN OF {text_emb.tolist()} LIMIT 3"
    result = session.execute(query)

    for row in result:
        print("Colour:", row.colour)
        print("Desc:", row.description)

def init():
    global session
    global cluster

    connect_astra()
    insert_data()

    st.set_page_config(
        page_title="Astra Vector Hybrid Search",
    )

    st.header("Astra Vector Hybrid Search :red[Example] :chart")
    st.markdown("Explore how AstraDB Vector and Hybrid is used to search across embedded data")

    #with st.sidebar:
    #    user_input = st.text_input("Your search based on VECTOR|TEXT|HYBRID, colour, description: ", key="user_input2")

    tab1, tab2 = st.tabs(["Vector Embeddings ", "Search", ])


    with tab1:
        st.markdown("## Images")
        df = pd.DataFrame(lst)
        st.dataframe(df)

    #message("Search for some dessert")

    with tab2:
        st.text('Your search based on :red [VECTOR|TEXT|HYBRID], red[colour], red[description]')
        user_input = st.text_input('Search for some dessert...')


    #Expecting in the format above
    if user_input:
        print("User entered: ", user_input)

        words = user_input.split(', ')
        # Check if there are exactly 3 elements after splitting
        if len(words) != 3:
            st.text("Input invalid type example:  VECTOR, red, cream cake", is_user=False)
            return

        # Extract individual fields
        data_type, colour, description = words

        # Check if data_type is valid
        data_type = data_type.strip().upper()  # Convert to uppercase for case-insensitive comparison
        if data_type not in ["VECTOR", "TEXT", "HYBRID"]:
            message("Input type, should be: VECTOR, TEXT or HYBRID", is_user=False)
            return

        if data_type == "VECTOR":
            print("Processing VECTOR data...")
            #vector_ann = ann_response(user_input)

            #vector_query_str = "round cake that's red"
            print("Query String: ", description)
            text_emb = img_model.encode(description)
            print(f"ANN model provided embeddings for the string: 'search': {text_emb.tolist()}")

            # Retrieve the nearest matching image from the database
            query = f"SELECT colour, description FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {text_emb.tolist()} LIMIT 3"
            print(query)

            result = session.execute(query)

            s = ''

            for row in result:
                print("Colour:", row.colour)
                print("Desc:", row.description)
                s += "- " + row.colour + " " + row.description + "\n"

            st.markdown(s)

        elif data_type == "TEXT":
            print("Processing TEXT data...")
            text_analyzer = text_response(user_input)
        else:
            print("Processing HYBRID data ..")
            hybrid = hybrid_response(user_input)


def image_embedding(image, model):
    return model.encode(image)

def main():
    init()

if __name__ == '__main__':
    main()
