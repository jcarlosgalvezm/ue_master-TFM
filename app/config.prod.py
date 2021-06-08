import dotenv
import os


PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir)

dotenv_path = os.path.join(PROJECT_DIR, '.env.prod')
dotenv.load_dotenv(dotenv_path)

## Cloudant
MODEL_CATALOG_URL = os.getenv('MODEL_CATALOG_URL')
MODEL_CATALOG_APIKEY = os.getenv('MODEL_CATALOG_APIKEY')

## COS (Cloud Object Storage)
COS_ENDPOINT = os.getenv('COS_ENDPOINT')
COS_API_KEY_ID = os.getenv('COS_API_KEY_ID')
COS_INSTANCE_CRN = os.getenv('COS_INSTANCE_CRN')
COS_BUCKET = os.getenv('COS_BUCKET')
