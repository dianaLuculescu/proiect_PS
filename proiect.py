import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from kneed import KneeLocator

st.markdown(
    """
    <style>
    .custom-title {
        color: #F39C12;
        font-size: 40px;
        text-align: center;
        color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

section = st.sidebar.radio("Navigați la:",
                           ["Prezentare generală",
                            "Analiză conținut Disney+",
                            "Grupare și agregare",
                            "Vizualizări grafice",
                            "Clusterizare",
                            "Regresie Logistica",
                            "Harta Disney+"])

date = pd.read_csv("disney_plus_titles.csv",index_col=0)

if section == "Prezentare generală":
    st.title("Disney Plus")
    st.header("Ce este Disney Plus?")
    st.markdown("""
    ## 🎬 Despre Disney Plus

    Disney Plus este un serviciu de streaming **video-on-demand** operat de **The Walt Disney Company**. Lansat oficial pe **12 noiembrie 2019**, platforma a devenit rapid unul dintre cei mai importanți jucători din industria streaming-ului, concurând direct cu Netflix, Amazon Prime Video și HBO Max.

    ### 🌟 Ce oferă Disney Plus?

    Serviciul include o colecție extinsă de filme, seriale și documentare, atât **producții originale**, cât și conținut clasic Disney. Printre cele mai populare francize disponibile pe platformă se numără:
    - **Disney** 🎠 – filme și seriale animate clasice, producții recente și remake-uri live-action.
    - **Pixar** 🎨 – animații apreciate la nivel global, cum ar fi *Toy Story, Finding Nemo, Inside Out*.
    - **Marvel** 🦸‍♂️ – universul cinematic Marvel, inclusiv seriale exclusive (*Loki, WandaVision, Moon Knight*).
    - **Star Wars** 🚀 – întreaga saga *Star Wars*, alături de producții originale precum *The Mandalorian* și *Ahsoka*.
    - **National Geographic** 🌍 – documentare despre știință, natură și istorie (*Free Solo, Welcome to Earth*).
    - **Star (în unele regiuni)** 🌟 – conținut destinat publicului matur, incluzând producții de la 20th Century Studios și FX.

    ### 📈 Creșterea și Expansiunea Disney Plus

    - **Lansare inițială:** 12 noiembrie 2019 în SUA, Canada și Olanda.
    - **Expansiune globală:** Serviciul este disponibil în peste **60 de țări**, inclusiv în Europa, Asia și America Latină.
    - **Abonați activi:** Până în 2024, Disney Plus a depășit pragul de **150 de milioane de abonați** la nivel mondial.
    - **Concurență:** Se află într-o competiție directă cu **Netflix, Prime Video, Apple TV+ și HBO Max**.

    ### 💰 Modele de Abonament

    Disney Plus oferă mai multe planuri de abonament, care pot varia în funcție de regiune:
    - **Plan standard:** Acces la întreaga bibliotecă, fără reclame.
    - **Plan cu reclame:** Disponibil în unele țări la un preț redus.
    - **Pachete combinate:** În SUA, Disney Plus oferă pachete cu **Hulu** și **ESPN+**.

    ### 🔮 Viitorul Disney Plus

    - **Extinderea conținutului original** – Disney investește masiv în noi producții Marvel și Star Wars.
    - **Adăugarea funcționalităților interactive** – Posibilitatea de vizionare în grup (*GroupWatch*).
    - **Lansări simultane în cinema și streaming** – Filme precum *Black Widow* și *Jungle Cruise* au fost lansate direct pe platformă.
    - **Dezvoltarea AI pentru recomandări personalizate** – Algoritmi îmbunătățiți pentru a oferi conținut relevant utilizatorilor.

    """)



elif section == "Analiză conținut Disney+":

    st.title("Analiză conținut Disney+")

    num_filme = date[date["type"] == "Movie"].shape[0]
    num_seriale = date[date["type"] == "TV Show"].shape[0]

    st.subheader("Distribuția conținutului pe platformă")
    st.write(f"🎬 **Număr total de filme:** {num_filme}")
    st.write(f"📺 **Număr total de seriale:** {num_seriale}")
    st.write("Mai jos puteți vizualiza setul de date utilizat pentru analiza conținutului Disney Plus.")

    nr_randuri = st.slider("Selectați câte rânduri doriți să fie afișate:", min_value=5, max_value=len(date), value=5,
                         step=5)

    st.dataframe(date.head(nr_randuri))


    date["country"].fillna("Unknown", inplace=True)
    date["director"].fillna("Not specified", inplace=True)
    date["cast"].fillna("Not specified", inplace=True)
    date["rating"].fillna("Unrated", inplace=True)
    date["date_added"].fillna("Unknown", inplace=True)

    limita_inferioara = 1940
    limita_superioara = 2025
    extreme = date[(date["release_year"] < limita_inferioara) | (date["release_year"] > limita_superioara)]

    if not extreme.empty:
        st.warning(
            f"S-au identificat {len(extreme)} valori extreme în `release_year` (în afara intervalului {limita_inferioara}-{limita_superioara}):")
        st.dataframe(extreme[["title", "release_year"]])

        elimina_extreme = st.checkbox("🔘 Elimină automat aceste valori extreme")

        if elimina_extreme:
            date = date[(date["release_year"] >= limita_inferioara) & (date["release_year"] <= limita_superioara)]
            st.success("✅ Valorile extreme au fost eliminate din dataset.")
    else:
        st.success("✅ Nu există valori extreme evidente în `release_year`.")

    st.subheader("🔡 Codificare a datelor")

    encoder = LabelEncoder()
    date["type_encoded"] = encoder.fit_transform(date["type"])

    st.write("🔍 Codificare coloanei `type` (Movie/TV Show) în valori numerice:")
    st.dataframe(date[["type", "type_encoded"]].drop_duplicates())

    st.markdown("📌 Codificare One-Hot pentru coloana `rating` (primele 5 rânduri):")
    encoder_ohe = OneHotEncoder(handle_unknown='ignore')  # fără sparse
    rating_encoded = encoder_ohe.fit_transform(date[["rating"]]).toarray()

    rating_df = pd.DataFrame(rating_encoded, columns=encoder_ohe.get_feature_names_out(["rating"]))
    st.dataframe(rating_df.head(5))

    st.subheader("🎭 Scalare: Numărul de genuri per titlu")

    date["genre_count"] = date["listed_in"].apply(lambda x: len(str(x).split(", ")))
    scaler_gen = StandardScaler()
    date["genre_count_scaled"] = scaler_gen.fit_transform(date[["genre_count"]])

    st.dataframe(date[["title", "genre_count", "genre_count_scaled"]].head(10))


elif section == "Grupare și agregare":
    st.title("📊 Grupare și agregare")

    optiuni_grupare = {
        "Anul lansării": "release_year",
        "Genul filmului": "listed_in",
        "Țara de producție": "country",
        "Tipul conținutului": "type"
    }

    optiune_selectata = st.sidebar.selectbox("Selectați criteriul de grupare:", list(optiuni_grupare.keys()))
    coloana_grupare = optiuni_grupare[optiune_selectata]

    grupare_date = date.groupby(coloana_grupare).size().reset_index(name="Număr de producții")
    grupare_date = grupare_date.sort_values("Număr de producții", ascending=False)

    st.subheader(f"📊 Distribuția conținutului după {optiune_selectata}")
    st.write(f"Top {min(10, len(grupare_date))} categorii în funcție de: numărul de producții:")
    st.dataframe(grupare_date.head(10))

    st.subheader("📈 Media anului de lansare în funcție de tipul conținutului")
    medii_tip = date.groupby("type")["release_year"].mean().reset_index()
    st.dataframe(medii_tip)

    st.subheader("🌍 Număr mediu de genuri per țară")
    date["genre_count"] = date["listed_in"].apply(lambda x: len(str(x).split(", ")))
    medii_genuri_tari = date.groupby("country")["genre_count"].mean().reset_index().sort_values(by="genre_count", ascending=False)
    st.dataframe(medii_genuri_tari.head(10))

    st.subheader("📅 Număr de producții pe an și tip de conținut")

    agregare_multipla = date.groupby(["release_year", "type"]).size().unstack().fillna(0).astype(int)

    toate_rindurile = st.checkbox("Afișează toti anii", value=False)

    if toate_rindurile:
        st.dataframe(agregare_multipla)
    else:
        st.dataframe(agregare_multipla.tail(10))  # Doar ultimii ani



elif section == "Vizualizări grafice":
    st.title(" Vizualizări Grafice")

    st.subheader("📅 Distribuția conținutului Disney+ pe ani")
    an_produse = date.groupby("release_year").size().reset_index(name="Număr de producții")
    an_produse = an_produse.sort_values("release_year", ascending=True)

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=an_produse, x="release_year", y="Număr de producții", marker="o", color="blue")
    plt.xlabel("Anul Lansării")
    plt.ylabel("Număr de Produse")
    plt.title("Evoluția Numărului de Produse Disney+ pe Ani")
    st.pyplot(plt)

    st.subheader("🎭 Top 10 genuri pe Disney+")
    top_genuri = date["listed_in"].str.split(", ").explode().value_counts().reset_index()
    top_genuri.columns = ["Gen", "Număr de producții"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_genuri.head(10), x="Număr de producții", y="Gen", palette="viridis")
    plt.xlabel("Număr de Produse")
    plt.ylabel("Gen")
    plt.title("Top 10 Genuri de Filme/Seriale pe Disney+")
    st.pyplot(plt)

    st.subheader("🎬 Distribuția Filmelor vs. Serialelor")
    tipuri_continut = date["type"].value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(tipuri_continut, labels=tipuri_continut.index, autopct="%1.1f%%", colors=["#ff9999","#66b3ff"], startangle=90)
    plt.title("Distribuția Filmelor și Serialelor pe Disney+")
    st.pyplot(plt)

elif section == "Clusterizare":

    st.title("Clusterizare cu KMeans – Disney+")

    date["genre_count"] = date["listed_in"].apply(lambda x: len(str(x).split(", ")))
    date["title_length"] = date["title"].apply(lambda x: len(str(x)))
    X = date[["genre_count", "title_length"]].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("📉 Determinarea numărului optim de clustere (Elbow Method)")

    inertia = []
    cluster_range = range(1, 11)
    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    kl = KneeLocator(cluster_range, inertia, curve="convex", direction="decreasing")
    k_opt = kl.elbow or 3

    fig, ax = plt.subplots()
    ax.plot(cluster_range, inertia, marker='o')
    ax.set_xlabel("Număr de clustere")
    ax.set_ylabel("Inertia")
    ax.set_title("Metoda Elbow – Alegerea numărului optim de clustere")
    ax.axvline(x=k_opt, color='red', linestyle='--', label=f"Optim: k = {k_opt}")
    ax.legend()
    st.pyplot(fig)

    st.success(f"🔍 Numărul optim de clustere determinat automat este: **{k_opt}**")

    nr_clusteri = st.slider("👉 Puteti selecta un alt număr de clustere:", 2, 10, k_opt)

    # KMeans cu numărul ales
    model = KMeans(n_clusters=nr_clusteri, random_state=42)
    clusteri = model.fit_predict(X_scaled)

    # PCA pentru vizualizare
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_cluster = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_cluster["Cluster"] = clusteri

    st.subheader("📊 Vizualizare Clustere (PCA)")
    fig2 = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_cluster, x="PC1", y="PC2", hue="Cluster", palette="Set2")
    plt.title(f"Vizualizare în 2D a celor {nr_clusteri} clustere")
    st.pyplot(fig2)

    st.markdown("📌 *Fiecare punct reprezintă un titlu Disney+, grupat pe baza numărului de genuri și lungimii titlului.*")

elif section == "Regresie Logistica":

    st.title("📈 Regresie Logistică – Clasificarea tipului de conținut Disney+")

    st.markdown("Prezicem dacă un titlu este **Movie** sau **TV Show**, pe baza caracteristicilor conținutului.")

    # Extragem caracteristici relevante
    date["genre_count"] = date["listed_in"].apply(lambda x: len(str(x).split(", ")))
    date["title_length"] = date["title"].apply(lambda x: len(str(x)))
    date["release_year"] = pd.to_numeric(date["release_year"], errors="coerce")
    df_model = date[["type", "genre_count", "title_length", "release_year"]].dropna()

    # Codificare target
    df_model["type_encoded"] = df_model["type"].apply(lambda x: 1 if x == "TV Show" else 0)
    X = df_model[["genre_count", "title_length", "release_year"]]
    y = df_model["type_encoded"]

    # Train/test split + scalare
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Acuratețe și raport
    acc = accuracy_score(y_test, y_pred)
    st.subheader("📈 Acuratețea modelului:")
    st.write(f"Acuratețea Logistic Regression: **{acc:.2f}**")

    st.subheader("📊 Raport de clasificare:")
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, target_names=["Movie", "TV Show"])).T
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))

    # Confusion matrix
    st.subheader("🧮 Matricea de confuzie:")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Movie", "TV Show"],
                yticklabels=["Movie", "TV Show"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📈 Curba ROC & AUC")
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle='--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Curba ROC")
    ax2.legend()
    st.pyplot(fig2)

    # Predicție pentru un titlu nou
    st.subheader("🔍 Testează un titlu nou:")
    genuri = st.slider("Număr genuri", 1, 5, 2)
    titlu_len = st.slider("Lungime titlu", 5, 50, 20)
    an = st.slider("Anul lansării", 1940, 2025, 2020)
    x_nou = scaler.transform([[genuri, titlu_len, an]])
    y_prob = model.predict_proba(x_nou)[0]
    pred = model.predict(x_nou)[0]
    eticheta = "TV Show" if pred == 1 else "Movie"
    st.write(f"🔮 Predicție: **{eticheta}** – Probabilitate: Movie: `{y_prob[0]:.2f}`, TV Show: `{y_prob[1]:.2f}`")

elif section == "Harta Disney+":
    st.title("🌍 Distribuția globală a producțiilor Disney+")

    tari_disney = date["country"].dropna().str.split(", ").explode().value_counts().reset_index()
    tari_disney.columns = ["country", "nr_productii"]

    # 2. Citim GeoJSON-ul cu granițele țărilor lumii
    geojson_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    world = gpd.read_file(geojson_url)

    tari_disney["country"] = tari_disney["country"].replace({
        "United States": "United States of America",
        "South Korea": "Republic of Korea",
        "Russia": "Russian Federation",
        "Vietnam": "Viet Nam",
        "Iran": "Iran (Islamic Republic of)",
        "Venezuela": "Venezuela (Bolivarian Republic of)",
        "Bolivia": "Bolivia (Plurinational State of)",
        "Tanzania": "United Republic of Tanzania",
        "Syria": "Syrian Arab Republic",
        "Moldova": "Republic of Moldova"
    })

    harta = world.merge(tari_disney, how="left", left_on="ADMIN", right_on="country")
    harta["nr_productii"] = harta["nr_productii"].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    harta.plot(column="nr_productii", cmap="Oranges", linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    ax.set_title("Număr de producții Disney+ pe tari", fontsize=16)
    ax.axis("off")
    st.pyplot(fig)