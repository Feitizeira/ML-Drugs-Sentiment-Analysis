import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def intro():

    st.image("data/portada.jpeg")
    st.title("ML Drugs Sentiment Analysis 📊 ")
    st.subheader("Mie Taira - Carmen Rey")

    st.sidebar.success("Analizando dataset")

    st.markdown(
        """
        - Nuestro dataset se ha publicado originalmente en [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29#)
        Citation: Felix Gräßer, Surya Kallumadi, Hagen Malberg, and Sebastian Zaunseder. 2018. Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning. In Proceedings of the 2018 International Conference on Digital Health (DH '18). ACM, New York, NY, USA, 121-125.
        
        - También hemos encontrado información disponible en [kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018) sobre el mismo dataset.

        - El conjunto de datos contiene revisiones de pacientes sobre medicamentos específicos junto con condiciones relacionadas y una calificación de paciente de 10 estrellas que refleja la satisfacción general del paciente. Los datos se obtuvieron rastreando páginas web de revisiones de productos farmacéuticos.
 
        - Se trata de 215.063 registros iniciales con 7 categorías, de los que finalmente haremos uso de **128.478 rows × 5 columns**
        """
    )

    st.subheader("Introducción al problema:")
    st.markdown(
        """
        El mundo del mercado farmacéutico americano es un tema muy contovertido pero no se puede negar que es **la mayor parte de la economia USA** y que ha salvado y mejorado la calidad de vida con sus medicamentos.

        •   Mercado farmaceuticos en USA ocupa **48% del nivel global** en 2020.

        •   La FDA aprueba de media de **38 drogas al año**.

        •   La FDA **solo aprueba el 12%** de todas las drogas que están en ensayo clinico.

        •   En 2020 el **41% de los americanos** cree que tienen mucha confianza en los laboratorios farmaceuticos. Creen que se preocupan por los intereses de los pacientes.

        •   Cada hogar gasta **442 dólares** al año en medicamentos.

        •   **Johnson & Johnson** es el labopratorio más grande del mundo por sus ingresos.
    
        •   En 2019, la industria farmacéutica americana gasta aproximadamente **83 billones de dólares** en ensayos y diseños de medicamentos.

        •   En 2025 USA prevé gastar de **605 a 635 billiones de dolares** en medicamentos.
        """
    )
    st.image("data/revenue.png")
    st.markdown(
        """
        •   En 2018, los medicamentos que más han sido ensayados eran para tratar el **cancer y enfermedades del sistema nervioso**.

        •   En 2020, **Humira** de laboratorios Abbvie es el más vendido en el mundo.

        •   **Lipitor** de laboratorios Pfizer es el más vendido en USA.

        •   Los medicamentos terapéuticos que más ingresos han generado en 2019 son los **oncológicos**.
        """
    )
    st.image("data/top5drugs.png")
    st.markdown(
        """
        •   Estos medicamentos para tratar el cancer generan aproximadamente **99,5 billones $** de ingresos.

        •   El segundo medicamento que más ingresos genera es para tratar la Diabetes, **67 billiones $** en el mismo año.

        •   Ninguna de estas clases tiene "el mayor número de recetas por medico". La categoría de medicamentos que más se receta es la **"hipertension"** pero solo generó **7,8 billiones de dolares** en 2019.

        •   La FDA prueba **53 nuevos medicamentos en 2020**.

        •   Cuesta aproximadamente **2.6 billiones de dolares** desarollar un nuevo medicamento.

        """
    )

    st.header("TRATAMIENTO DE DATOS:")
    st.subheader("EDA: Exploratory Data Analysis")
    st.markdown(
        """
        •	🔢 Convertimos los floats con un decimal a **integers**.

        •	❌ Eliminamos los valores duplicados en el índice, **reseteamos el index**.

        •	❌ Eliminamos los valores **duplicados en las reviews 40%**. Esto fue particularmente lo más difícil de detectar porque no llamaba la atención, sólo al ver el análisis de sentimientos encontrábamos la misma frase repetida.

        •	📥 Rellenamos los **NaN** de la columna condición con "EMPTY" porque sí había información relevante sobre las medicaciones aúnque el usuario no registró una categoría.

        •	❌ Eliminamos 2 columnas que no aportan información para nuestro estudio (la anonimización del autor de la review y la fecha de la review).

        •	😊 Este aspecto del proyecto no ha resultado complejo a pesar de invertir mucho **tiempo** en ello. Por una vez no nos ha generado incertidumbre .

        •	📈 Hemos tenido que descartar analizar todos nuestros sub-datasets porque el NaiveBayesAnalyzer hubiese tardado ~84 horas, para obtener resultados poco significativos. Por tanto redujimos el conjunto de datos a sólo 900 entradas para este análisis. Sin embargo el SentimentIntensityAnalyzer trabaja mucho más rápido y sí pudimos analizar todos los sub-datasets (~90.000 reviews) en apenas 2 minutos.
        """
    )

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')
    st.dataframe(df, width = 900, height = 400, use_container_width = True)

def diabetes():
    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    df_diabetes = df[ (df["condition"] == "Diabetes, Type 1") 
                    | (df["condition"] == "Diabetes, Type 2") 
                    | (df["condition"] == 'Gestational Diabetes')
                    | (df["condition"] == 'Diabetes Insipidus')].sort_values("rating", ascending = False)
    st.dataframe(df_diabetes)

    fig1 = px.sunburst(data_frame = df_diabetes,
                values     = "rating",
                title      = "Valoración de los medicamentos utilizados para curar la Diabetes",
                width      = 650,
                height     = 650,
                path       = ["rating", "condition", "drugName"],
                hover_name = "drugName",
                color      = "rating")
    st.plotly_chart(fig1)

    fig5 = px.treemap(data_frame = df_diabetes,
            values     = "rating",
            path       = ["condition","rating", "drugName"],
            hover_name = "drugName",
            color      = "drugName")
    st.plotly_chart(fig5)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_diabetes.png")

    st.write("**Liraglutide**: Droga más valorada en Diabetes, type 2")
    st.image("data/df_sentsia_lira.png")
    st.image("data/df_sentNB_lira.png")
    st.image("data/pol_lira.png")


# ### DFS CON LAS VALORACIONES DE CADA ENFERMEDAD/CONDICIÓN
def cancer():

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    listacancer = ['Multiple Myeloma', 'Malignant Glioma','Lymphoma',"Hodgkin's Lymphoma", 
                'Multiple Endocrine Adenomas', 'Anaplastic Astrocytoma', 'llicular Lymphoma', 
                'Peripheral T-cell Lymphoma','Glioblastoma Multi', 'Melanoma', 
                'Squamous Cell Carcinoma', 'Cutaneous T-cell Lymphoma', 
                'Osteolytic Bone Lesions of Multiple Myeloma', 'Melanoma, Metastatic', 
                'Renal Cell Carcinoma', 'Hepatocellular Carcinoma', 'Soft Tissue Sarcoma', 
                'Hemangioma', 'Mantle Cell Lymphoma', "Non-Hodgkin's Lymphoma" , 
                'Anaplastic Oligodendroglioma', 'Glioblastoma Multiforme', 
                'Basal Cell Carcinoma','Salivary Gland Cance','Stomach Cance',
                'Ovarian Cance','Head and Neck Cance','Thyroid Cance','Endometrial Cance',
                'Skin Cance','Pancreatic Cance', 'Colorectal Cance','Gastric Cance', 'Testicular Cance',
                'Breast Cancer, Prevention', 'Breast Cancer, Metastatic', 'Breast Cancer, Palliative', 
                'Breast Cancer, Adjuvant' ]
    listadfs = []
    for canc in listacancer:
        listadfs.append(df[df["condition"]==canc])  
    df_cancer = pd.concat(listadfs, axis = 0)    
    st.dataframe(df_cancer)

    fig2 = px.sunburst(data_frame = df_cancer,
                values     = "rating",
                title      = "Valoración de los medicamentos utilizados para curar el Cancer",
                width      = 650,
                height     = 650,
                path       = [ "rating", "condition","drugName"],
                hover_name = "drugName",
                color      = "rating")
    st.plotly_chart(fig2)

    fig6 = px.treemap(data_frame = df_cancer,
            values     = "rating",
            path       = ["condition","rating", "drugName"],
            hover_name = "drugName",
            color      = "drugName")
    st.plotly_chart(fig6)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_cancer.png")
    st.image("data/df_sentNB_cancer.png")
    st.image("data/pol_cancer.png")

    st.write("Pazopanib: Droga más valorada en Renal Cell Carcinoma")
    st.image("data/df_sentsia_pazo.png")
    st.image("data/df_sentNB_pazo.png")
    st.image("data/pol_pazo.png")

def depresion():

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    df_depresion = df[df["condition"]=="Depression"]
    st.dataframe(df_depresion)

    fig3 = px.sunburst(data_frame = df_depresion,
            title      = "Valoración de los medicamentos utilizados para curar la Depresión",
            width      = 650,
            height     = 650,
            path       = ["condition", "rating", "drugName"],
            hover_name = "drugName",
            color      = "rating")
    st.plotly_chart(fig3)

    fig7 = px.treemap(data_frame = df_depresion,
            values     = "rating",
            path       = ["condition","rating", "drugName"],
            hover_name = "drugName",
            color      = "drugName")

    st.plotly_chart(fig7)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_depresion.png")

    st.write("Bupropion: Droga más valorada en Depression")
    st.image("data/df_sentsia_bupro.png")

def control():

    st.markdown(f"# {list(page_names_to_funcs.keys())[4]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    df_control = df[df["condition"]=="Birth Control"]
    st.dataframe(df_control)
    
    fig4 = px.sunburst(data_frame = df_control,
                title      = "Valoración de los medicamentos utilizados para el control de Natalidad",
                width      = 650,
                height     = 650,
                path       = ["condition", "rating", "drugName"],
                hover_name = "drugName",
                color      = "rating")
    st.plotly_chart(fig4)

    fig8 = px.treemap(data_frame = df_control,
            values     = "rating",
            path       = ["condition","rating", "drugName"],
            hover_name = "drugName",
            color      = "drugName")
    st.plotly_chart(fig8)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_controlnatalidad.png")

    st.write("Etonogestrel: Droga más valorada en birth control")
    st.image("data/df_sentsia_etono.png")

def humira():

    st.markdown(f"# {list(page_names_to_funcs.keys())[5]}")
    
    st.image("data/humira-foto.jpg", width = 300)
    st.subheader("Medicamento más vendido en el mundo")
    st.write("Humira de Abbvie es el fármaco más vendido en el mundo en 2020. Se usa para tratar: Artritis reumatoide , Psoriasis en placas, Hidradenitis supurativa, Enfermedad de Crohn, Colitis ulcerosa, Uveitis no infecciosa, Artritis reumatoide, Psoriasis en placas, Hidradenitis supurativa, Enfermedad de Crohn, Colitis ulcerosa, Uveitis no infecciosa")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    humira= df[df["drugName"]=="Humira"]
    st.dataframe(humira)
    
    st.subheader("Resultados obtenidos")
    st.write("Media de las valoraciones:", round(humira["rating"].mean()))
    st.write("Suma de comentarios útiles:", humira["usefulCount"].sum())
    st.image("data/df_sentsia_humira.png")

def lipitor():

    st.markdown(f"# {list(page_names_to_funcs.keys())[6]}")
    st.image("data/Lipitor-foto.png", width = 400)
    st.subheader("Medicamento más vendido en US")
    st.write("Lipitor de Pfizer es el fármaco de mayor venta en general en los EE. UU. Se utiliza para el control de los niveles altos de colesterol (Hypercholesterolemia - Atorvastatina)")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    lipi= df[df["drugName"]=="Lipitor"]
    st.dataframe(lipi)

    st.subheader("Resultados obtenidos")
    st.write("Media de las valoraciones:", round(lipi["rating"].mean()))
    st.write("Suma de comentarios útiles:", lipi["usefulCount"].sum())

    st.image("data/df_sentsia_lipitor.png")
    st.image("data/df_sentNB_lipitor.png")
    st.image("data/pol_lipi.png")

def keytruda():

    st.markdown(f"# {list(page_names_to_funcs.keys())[7]}")
    
    st.image("data/keytruda.jpeg", width = 300)
    st.subheader("2º medicamento más vendido en US")
    st.write("Es un tipo de anticuerpo monoclonal inhibidor de puntos de control inmunitario. También se llama pembrolizumab. Se une a una proteína para ayudar a las células inmunitarias a destruir más células cancerosas, y que se usa para el tratamiento de muchos tipos diferentes de cáncer")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    st.write("Keytruda: ")
    keytruda= df[df["drugName"]=="Keytruda"]
    st.dataframe(keytruda)

    st.subheader("Resultados obtenidos")
    st.write("Media de las valoraciones:", round(keytruda["rating"].mean()))
    st.write("Suma de comentarios útiles:", keytruda["usefulCount"].sum())

    st.image("data/df_sentsia_keytruda.png")
    st.image("data/df_sentNB_keytruda.png")
    st.image("data/pol_keytruda.png")

def low_high_rates():

    st.markdown(f"# {list(page_names_to_funcs.keys())[8]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    st.subheader("Valoraciones más bajas: Rating = 1")
    df_low_rates = df[(df["rating"] == 1.0)]
    st.dataframe(df_low_rates)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_low_rates.png")
    
    st.subheader("Valoraciones más altas: Rating = 10")
    df_high_rates = df[(df["rating"] == 10.0)]
    st.dataframe(df_high_rates)

    st.subheader("Resultados obtenidos")
    st.image("data/df_sentsia_high_rates.png")
   

def visualizaciones():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(f"# {list(page_names_to_funcs.keys())[9]}")

    train = pd.read_csv('data/drugsComTrain_raw.tsv',sep ="\t")
    test = pd.read_csv('data/drugsComTest_raw.tsv',sep ="\t")
    df = pd.concat([train, test])
    # EDA
    df.rating = df.rating.astype(int)
    df.reset_index(inplace = True, drop = True)
    df.drop_duplicates(subset=["review"], inplace = True)
    df.condition.fillna("EMPTY", inplace = True)
    df.drop(columns = ["Unnamed: 0", "date"], inplace = True)
    df["review"] = df["review"].str.replace('&#039;', '')
    df["review"] = df["review"].str.replace("&amp;", '')
    df["review"] = df["review"].str.replace('\r\n', '')
    df["review"] = df["review"].str.replace('&quot;', '')
    df["review"] = df["review"].str.replace("'" , '')
    df["review"] = df["review"].str.replace("`" , '')
    df["review"] = df["review"].str.replace('"' , '')
    df["review"] = df["review"].str.replace('-' , '')

    comentarios_utiles = df.sort_values("usefulCount", ascending = False)

    # Los comentarios más útiles,
   
    grafico = sns.barplot(  x = comentarios_utiles.drugName[0:15], 
                            y = comentarios_utiles.usefulCount[0:15], 
                            palette="Set1")
    grafico.set_title('Los 10 medicamentos con mejor recuento de "comentario útil"')
    grafico.set_ylabel("Número de comentarios útiles")
    grafico.set_xlabel("Nombre del medicamento")
   
    plt.setp(grafico.get_xticklabels(), rotation=90);
    st.pyplot()

    st.markdown("""
    - Sertraline se usa para el tratamiento del trastorno depresivo mayor, trastorno obsesivo-compulsivo (OCD, por sus siglas en inglés), trastorno de pánico, trastorno de ansiedad social (SAD, por sus siglas en inglés), y trastorno de estrés postraumático (PTSD, por sus siglas en inglés).
    - Mirena es un dispositivo intrauterino (DIU) hormonal que puede proporcionar control de la natalidad a largo plazo (anticonceptivo).
    - Sertraline (nombre genérico de Zoloft) es un inhibidor selectivo de la recaptación de serotonina (SSRI, por sus siglas en inglés). Sertraline se usa para el tratamiento del trastorno depresivo mayor, trastorno obsesivo-compulsivo (OCD, por sus siglas en inglés), trastorno de pánico, trastorno de ansiedad social (SAD, por sus siglas en inglés), y trastorno de estrés postraumático (PTSD, por sus siglas en inglés). Sertraline también puede utilizarse para tratar el trastorno disfórico premenstrual.
    """)

    # Las 10 mejores drogas

    sns.set(font_scale = 1.2, style = 'darkgrid')
    plt.rcParams['figure.figsize'] = [15, 6]

    rating = dict(df.loc[df.rating == 10, "drugName"].value_counts())
    drugname = list(rating.keys())
    drug_rating = list(rating.values())

    sns_rating = sns.barplot(x = drugname[0:10], 
                             y = drug_rating[0:10], 
                             palette="Set2",
                            )

    sns_rating.set_title('Los 10 medicamentos con mejor valoración')
    sns_rating.set_ylabel("Número de valoraciones")
    sns_rating.set_xlabel("Nombre del medicamento")
    plt.setp(sns_rating.get_xticklabels(), rotation=90);
    st.pyplot()
    
    st.markdown("""
    - Levonorgestrel es un anticonceptivo oral de urgencia, también conocida como pastilla del día después. 
    - Phentermine se usa junto con la dieta y los ejercicios para tratar la obesidad, especialmente en las personas con factores de riesgo como presión arterial alta, colesterol elevado, o diabetes.
    - Etonogestrel es una progestina esteroidea que se utiliza como anticonceptivo hormonal, cuyo uso más notable es en los implantes subdérmicos y en los anillos vaginales.
    """)

    # Las 10 condiciones que más afectan a los pacientes

    cond = dict(df['condition'].value_counts())
    top_condition = list(cond.keys())[0:10]
    values = list(cond.values())[0:10]
    sns.set(style = 'darkgrid', font_scale = 1.3)
    plt.rcParams['figure.figsize'] = [15, 6]

    sns_ = sns.barplot(x = top_condition, y = values, palette = 'rainbow')
    sns_.set_title("Las 10 condiciones/enfermedades con más datos")
    sns_.set_xlabel("Condiciones/Enfermedades")
    sns_.set_ylabel("Número de valoraciones")
    plt.setp(sns_.get_xticklabels(), rotation=90);
    st.pyplot()

    st.markdown(
        """ 
        #### Ejemplos de mal funcionamiento de la valoración de sentimientos
•	🤕 pain so intense sent me into depression for 3 months.

Sentiment(classification='pos', p_pos=0.9320985700567367, **p_neg=0.06790142994326284**)

•	😷 he also coughs constantly and brings up white foamy phlegm.

Sentiment(classification='pos', p_pos=0.9184164520334995, **p_neg=0.08158354796650012**)

#### Ejemplos de buen funcionamiento de la valoración de sentimientos
•	🤢 i am nauseous when i take the medicine.

Sentiment(classification='neg', p_pos=0.07359249251501992, **p_neg=0.9264075074849801**)

•	😀 i have a fantastic doctor) ive had ease of access, when i mention ibrance to medical professionals and the like their response is positive according to what theyre hearing.

Sentiment(classification='pos', **p_pos=0.9312284705056294**, p_neg=0.06877152949436723)
        """
    )

def conclusion():

    st.markdown(f"# {list(page_names_to_funcs.keys())[10]}")

    st.image("data/nlp.jpg" , width = 500)
    
    st.markdown(
        """
        • La conclusión de nuestro estudio sobre el procesamiento de lenguaje natural de aprendizaje automático de conjuntos de datos de drogas es que **estos algoritmos deben mejorarse**. Realmente no obtenemos una conclusión clara sobre las revisiones de los medicamentos. Incluso cuando filtramos las reseñas por las calificaciones más altas, no encontramos diferencias significativas.

        • La industria farmacéutica es un **gigante mundial y desempeña un papel importante en la economía de los EEUU**, desde la generación de ingresos hasta la creación y el mantenimiento de puestos de trabajo.

        • Los consumidores pueden ver esto a través de los anuncios y los medicamentos que compran, pero lo que no ven es la cantidad de **investigación y desarrollo** que se dedica a crear los medicamentos que compran.

        • Las empresas pagan a los empleados para que creen los compuestos, realicen extensas pruebas clínicas y continúen mejorando el fármaco con el tiempo. La FDA se asegura de que todas las pruebas se realicen correctamente, siguiendo **buenas prácticas clínicas** y puede poner su sello de aprobación para que los medicamentos lleguen al mercado.

        • El mundo farmacéutico siempre se está adaptando a medida que cambian los problemas de salud, **se establecen nuevas y mejores prácticas y mejora la tecnología**s. Tiene el poder de mejorar o dañar directamente una gran cantidad de vidas, lo que la convierte en una de las industrias más complejas y fascinantes.
    """)
    # a pie chart to represent the sentiments of the patients

    size = [612, 388]
    colors = ['orange', 'lightgreen']
    labels = "Positive Sentiment","Negative Sentiment"
    explode = [0, 0.02]

    plt.rcParams['figure.figsize'] = (1.7, 1.7)
    plt.pie(size, 
            colors = colors, 
            labels = labels, 
            explode = explode, 
            autopct = '%.2f%%', 
            textprops = {'size': '5'} )
    plt.axis('off')
    plt.title('Pie Chart Representation of Sentiments', fontsize = 6)
    plt.show()
    st.pyplot()
    
    #button
    if st.button('and this is... '):
        st.image("data/end.jpg")
    else:
        st.write(' ')
        

page_names_to_funcs = {
    "—"                         : intro,
    "Diabetes"                  : diabetes,
    "Cancer"                    : cancer,
    "Depresion"                 : depresion,
    "Control de Natalidad"      : control,
    "Humira"                    : humira,
    "Lipitor"                   : lipitor,
    "Keytruda"                  : keytruda,
    "Low & High Ratings"        : low_high_rates,
    "Visualizaciones"           : visualizaciones,
    "Conclusion"                : conclusion,
}

drugs = st.sidebar.selectbox("Choose an option:", page_names_to_funcs.keys())
page_names_to_funcs[drugs]()