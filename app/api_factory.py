import io
import numpy as np
import pandas as pd
import pickle

from flask import current_app
from flask_restx import Namespace, Resource
from src.vendor.IBM import cloudant, cos
from src.data.make_dataset import get_dataset


def make_namespaces(api):

    cloudant_client = cloudant.get_client('MODEL_CATALOG')
    cos_client = cos.get_client()
    bucket = current_app.config['COS_MODEL_STORAGE_BUCKET']
    db = current_app.config['MODEL_CATALOG_DB']
    predictions_db = current_app.config['MODEL_PREDICTIONS_DB']

    model_definition = cloudant_client.get_document(db=db,
        doc_id='ds_jobs', revs=True).get_result()

    for num, rev in enumerate(
            model_definition['_revisions']['ids'][::-1], start=1
            ):
        try:
            fname = f'model_v{num}.pkl'
            ns = Namespace(f'v{num}', description=f'v{num}')
            upload_parser = ns.parser()
            upload_parser.add_argument('horas_formacion', type=int,
                required=True)
            upload_parser.add_argument('indice_desarrollo_ciudad', type=float,
                required=True)
            upload_parser.add_argument('ultimo_nuevo_trabajo', choices=[
                '1', '2', '3', '4', '>4', 'never'
                ], type=str, required=True)
            upload_parser.add_argument('tamano_compania', choices=[
                '<10', '10/49', '50-99', '100-500', '500-999', '1000-4999',
                '5000-9999', '10000+'
                ], type=str, required=True)
            upload_parser.add_argument('experiencia', choices=[
                '<1', '1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12',
                '13', '14', '15', '16', '17', '18', '19', '20', '>20'
                ], type=str, required=True)
            upload_parser.add_argument('educacion', choices=[
                'STEM', 'Humanities', 'Other', 'Business Degree', 'Arts',
                'No Major'
                ], type=str, required=True)
            upload_parser.add_argument('nivel_educacion', choices=[
                'Graduate', 'Masters', 'High School', 'Phd', 'Primary School'
                ], type=str, required=True)
            upload_parser.add_argument('experiencia_relevante', choices=[
                'Has relevent experience', 'No relevent experience'
                ], type=str, required=True)
            upload_parser.add_argument('tipo_compania', choices=[
                'Pvt Ltd', 'Funded Startup', 'Public Sector',
                'Early Stage Startup', 'NGO', 'Other',
                ], type=str, required=True)
            upload_parser.add_argument('genero', choices=[
                'Male', 'Female'
                ], required=True)
            upload_parser.add_argument('ciudad', required=True, type=str)

            cos_client.get_object(Bucket=bucket, Key=fname)

            @ns.route('/info')
            class Info(Resource):

                FNAME = fname
                BUCKET = bucket
                REVISION = f'{num}-{rev}'

                def get(self):
                    '''Model info'''
                    X_train, X_test, y_train, y_test = get_dataset()

                    model_buf = io.BytesIO()
                    cos_client.download_fileobj(Bucket=self.BUCKET,
                        Key=self.FNAME, Fileobj=model_buf)
                    model_buf.seek(0)
                    model = pickle.load(model_buf)
                    score = model.score(X_test, y_test)
                    return {
                        'definition': cloudant_client.get_document(
                                db=db,
                                doc_id='ds_jobs',
                                rev=self.REVISION).get_result(),
                        'score': score
                        }

            @ns.route('/predict')
            @ns.expect(upload_parser)
            class Predict(Resource):

                FNAME = fname
                BUCKET = bucket

                def get(self):
                    '''Predict employee'''
                    args = upload_parser.parse_args()
                    model_buf = io.BytesIO()
                    cos_client.download_fileobj(Bucket=self.BUCKET,
                        Key=self.FNAME, Fileobj=model_buf)
                    model_buf.seek(0)
                    model = pickle.load(model_buf)
                    X = pd.DataFrame([args])
                    X['universidad_matriculado'] = np.nan
                    X = X[[
                        'ciudad',
                        'indice_desarrollo_ciudad',
                        'genero',
                        'experiencia_relevante',
                        'universidad_matriculado',
                        'nivel_educacion',
                        'educacion',
                        'experiencia',
                        'tamano_compania',
                        'tipo_compania',
                        'ultimo_nuevo_trabajo',
                        'horas_formacion',
                        ]]
                    y_hat = model.predict(X)

                    doc = dict(**args, target=y_hat, model_version=num)
                    cloudant_client.post_document(
                        db=predictions_db,
                        document=doc
                    )

                    return {
                        'result': y_hat[0]
                    }

            yield ns

        except cos_client.exceptions.NoSuchKey:
            pass
