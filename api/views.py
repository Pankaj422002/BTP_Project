from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import json

# MU-RAG imports : 
import os, fitz, base64, qdrant_client, json, shutil
from PIL import Image, ImageDraw
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings, ServiceContext, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.schema import ImageDocument, ImageNode
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from IPython.display import HTML, display
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core import get_response_synthesizer


GOOGLE_API_KEY = "AIzaSyDzQpjlBre87IO1uRQZ5293ERzBZsQG14U"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class OURRAG:
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.file_name_with_extension = file_name+'.pdf'
        self.gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")
        self.index_img = None

        #check whether quadrant_db is present or not : 
        folder_path = "C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/qdrant_db3"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"The folder '{folder_path}' has been deleted.")
        else:
            print(f"The folder '{folder_path}' does not exist.")


    def _generated_page_image(self):

        pdf_file = 'C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/data2/'+self.file_name_with_extension
        output_directory_path = 'C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/' + self.file_name

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        # Open the PDF file
        pdf_document = fitz.open(pdf_file)

        # Iterate through each page and convert to an image
        for page_number in range(pdf_document.page_count):
            # Get the page
            page = pdf_document[page_number]

            # Convert the page to an image
            pix = page.get_pixmap()

            # Create a Pillow Image object from the pixmap
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Save the image
            image.save(f"./{output_directory_path}/page_{page_number + 1}.png")

        # Close the PDF file
        pdf_document.close()

    def get_img_summary(self, img_doc, prompt):
        response = self.gemini_pro.complete(
            prompt=prompt,
            image_documents=[img_doc],
        )
        return response

    def get_encoded_image(self, path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _generate_img_summaries(self):
        path = 'C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/' + self.file_name

        # img_list = []
        # image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        You have to generate summary only from the input images and read image in detail and depth. \
        Images may contain the ratings(in stars), companies log(s) so while generating summary pay extra attention in that part of image where rating and companies log(s) is mentioned and count the stars accurately because rating must be precise and correct. \
        When you are generating summary try to divide the image in segments and then generate summary in detail and depth. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a detailed and concise summary of the image that is well optimized for retrieval.\
        """ 

        documents_images_v2 = SimpleDirectoryReader(path).load_data()
        for i,val in enumerate(documents_images_v2):
            img_base64 = self.get_encoded_image(val.image_path)
            val.image_url = f"data:image/png;base64,{img_base64}"
            val.metadata['file_type']="data:image/png;base64,"
            
            summary = self.get_img_summary(val,prompt)
            val.text = summary.text
            # image_summaries.append(summary)
            # img_list.append(img_base64)
            documents_images_v2[i]=val
            
        return documents_images_v2 # img_list, image_summaries, 

    def _get_text_document(self):
        my_text_documents = SimpleDirectoryReader("C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/data2").load_data()
        return my_text_documents

    def ret_img_document(self):
        load_previous_evaluations=True
        if not load_previous_evaluations:
            img_documents = self._generate_img_summaries()
            # img_list, image_summaries, img_documents = generate_img_summaries("./pankj/")
            with open("C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/calculated_data/img_documents", 'w') as file:
                json.dump([vars(doc) for doc in img_documents], file)
        else:
            with open("C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/calculated_data/img_documents", 'r') as file:
                json_data_img_doc = json.load(file)

            # Create ImageDocument instances from the data
            img_documents = [ImageDocument(**doc) for doc in json_data_img_doc]

        return img_documents
    
    def ret_text_document(self):
        load_previous_evaluations=True
        if not load_previous_evaluations:
            my_text_documents = self._get_text_document()
            # img_list, image_summaries, img_documents = generate_img_summaries("./pankj/")
            with open("C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/calculated_data/my_text_documents", 'w') as file:
                json.dump([vars(doc) for doc in my_text_documents], file)
        else:
            with open("C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/calculated_data/my_text_documents", 'r') as file:
                json_data_txt_doc = json.load(file)

            # Create ImageDocument instances from the data
            my_text_documents = [Document(**doc) for doc in json_data_txt_doc]

        return my_text_documents


    def _get_index(self):
        self._generated_page_image()

        # storage context: 
        client = qdrant_client.QdrantClient(path="C:/Users/Saurabh Joshi/Desktop/BTP_Project/BTP_Research/api/qdrant_db3")
        text_store = QdrantVectorStore(
            client=client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client, collection_name="image_collection"
        )
        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )

        # service context: 
        embed_model = GeminiEmbedding(
            model_name="models/embedding-001", api_key=GOOGLE_API_KEY
        )
        service_context = ServiceContext.from_defaults(llm=Gemini(model_name="models/gemini-pro-vision"), embed_model=embed_model)
        
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001", api_key=GOOGLE_API_KEY
        )
        Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

        # generating mix docs: 
        img_documents = self.ret_img_document()
        my_text_documents = self.ret_text_document()

        mixed_doc = []
        for i in img_documents:
            mixed_doc.append(i)
        for i,val in enumerate(my_text_documents):
            val.metadata["id_key"]="thisisidkey_"+str(i)
            mixed_doc.append(val)

        index_img = MultiModalVectorStoreIndex.from_documents(
            mixed_doc,
            storage_context=storage_context,
            service_context=service_context
        )
        self.index_img = index_img
        return index_img

    def plt_img_base64(self, img_base64):
        """Disply base64 encoded string as image"""
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/png;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        display(HTML(image_html))

    def get_retrieve_result_1(self, query):
        prompt = (
        "Give a detailed and most accurate results that is well optimized for retrieval. \n\n"
        f"user question is : {query}\n"
        "Answer is : "
        ) 

        # index_img = self._get_index()
        index_img = self.index_img

        retriever = index_img.as_retriever(similarity_top_k=2, image_similarity_top_k=2)
        retrieval_results = retriever.retrieve(prompt)
        return retrieval_results


    #this is function is for user use : which help in showing the retrieved chunks of the user query: 
    def print_retrieved_chunks(self, query):

        res = self.get_retrieve_result_1(query)
        retrieved_image = []
        for res_node in res:
            if isinstance(res_node.node, ImageNode):
                self.plt_img_base64(res_node.node.image_url[22:])
            else:
                print(res_node.node.text, end="\n\n\n")


    def get_retrieve_result(self, query):
        # index_img = self._get_index()

        index_img = self.index_img

        retriever = index_img.as_retriever(similarity_top_k=3, image_similarity_top_k=3)

        # # defining prompt=====================================================prompt============
        template = (
            "We have provided context information below. \n"
            "The context information that is provide to you contains textdocument as well as imagedocument, and also imagedocument contains the image as well it's summary attached inside it's text field. \n"
            "context information is below: "
            "---------------------\n"
            "{context_str}"
            
            "\n---------------------\n"
            "You are financial analyst tasking with providing investment advice.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "You also have to analyse provided images and texts in detail before giving answer. \n"
            "Use this information to provide investment advice related to the user question and donot use any other information only use the provided information. \n"
            "User-provided question: {query_str} \n\n" 
        )
        
        qa_template = PromptTemplate(template)

        # # using prompt =============================================== using============================
        synth = get_response_synthesizer(
            text_qa_template=qa_template,    #, refine_template=custom_refine_prompt
        )
        
        query_engine = RetrieverQueryEngine(retriever, response_synthesizer=synth)

        response = query_engine.query(query)
        return response


murag_obj = OURRAG('dataset_radarview')
murag_obj._get_index()

# Create your views here.
class ChatApiView(APIView):
    def post(self, request):
        req_json = json.loads(request.body)
        query = req_json['query']
        
        res = murag_obj.get_retrieve_result(query)

        return Response({'data':res.response})

