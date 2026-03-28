import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from fpdf import FPDF
import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# ── Configuration page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="ETL Automatique",
    page_icon="🔧",
    layout="wide"
)

# ── Style CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: bold;
        color: #E50914; text-align: center; margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center; color: #666; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

# ── Titre principal ────────────────────────────────────────────────────
st.markdown('<div class="main-title">Pipeline ETL Automatique</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload ton dataset — le pipeline fait tout automatiquement</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CLASSE ETL
# ══════════════════════════════════════════════════════════════════════
class ETLAuto:
    def __init__(self, df, filename='dataset'):
        self.df_raw   = df.copy()
        self.df       = df.copy()
        self.filename = filename
        self.rapport  = []

    def audit(self):
        missing     = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        return pd.DataFrame({
            'Manquants'    : missing,
            'Pourcentage %': missing_pct,
            'Type'         : self.df.dtypes
        }).sort_values('Manquants', ascending=False)

    def transform(self):
    print('\n=== TRANSFORMATIONS ===')

    # T1 — Suppression doublons
    before = len(self.df)
    self.df.drop_duplicates(inplace=True)
    n = before - len(self.df)
    print(f'   T1 Doublons supprimes       : {n}')
    self.rapport.append(f'T1 Doublons : {n} supprimes')

    # T2 — Suppression colonnes ID inutiles
    id_cols = [c for c in self.df.columns
               if c.lower() in ['id', 'sale_id', 'index', 'unnamed: 0']]
    if id_cols:
        self.df.drop(columns=id_cols, inplace=True)
        print(f'   T2 Colonnes ID supprimees    : {id_cols}')
        self.rapport.append(f'T2 Colonnes ID supprimees : {id_cols}')
    else:
        print(f'   T2 Aucune colonne ID detectee')

    # T3 — Nettoyage espaces
    str_cols = self.df.select_dtypes(include='object').columns
    for col in str_cols:
        self.df[col] = self.df[col].astype(str).str.strip()
    print(f'   T3 Espaces nettoyes          : {len(str_cols)} colonnes')
    self.rapport.append(f'T3 Espaces : {len(str_cols)} colonnes nettoyees')

    # T4 — Imputation automatique
    n_total = 0
    for col in self.df.columns:
        n_miss  = self.df[col].isin(['nan','NaN','None','']).sum()
        n_miss += self.df[col].isnull().sum()
        if n_miss > 0:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].replace(
                    {'nan':'Unknown','NaN':'Unknown',
                     'None':'Unknown','':'Unknown'}
                ).fillna('Unknown')
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            n_total += n_miss
    print(f'   T4 Valeurs imputees          : {n_total}')
    self.rapport.append(f'T4 Imputation : {n_total} valeurs')

    # T5 — Conversion dates COMPLETE (6 features)
    n_dates = 0
    for col in self.df.select_dtypes(include='object').columns:
        if any(k in col.lower() for k in ['date', 'time']):
            try:
                self.df[col] = pd.to_datetime(
                    self.df[col], dayfirst=True, errors='coerce'
                )
                # Extraction des 6 features temporelles
                prefix = col.lower().replace('date', '').strip('_') or 'sale'
                self.df[f'year_{prefix}']      = self.df[col].dt.year.astype('Int64')
                self.df[f'month_{prefix}']     = self.df[col].dt.month.astype('Int64')
                self.df[f'day_{prefix}']       = self.df[col].dt.day.astype('Int64')
                self.df[f'quarter_{prefix}']   = self.df[col].dt.quarter.astype('Int64')
                self.df[f'month_name_{prefix}']= self.df[col].dt.strftime('%B')
                self.df[f'day_of_week_{prefix}']= self.df[col].dt.strftime('%A')
                n_dates += 1
            except:
                pass
    print(f'   T5 Dates converties          : {n_dates} (6 features chacune)')
    self.rapport.append(f'T5 Dates : {n_dates} colonnes converties en 6 features')

    # T6 — Outliers (colonnes numeriques uniquement)
    num_cols   = self.df.select_dtypes(include=np.number).columns
    n_outliers = 0
    for col in num_cols:
        Q1  = self.df[col].quantile(0.25)
        Q3  = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = ((self.df[col] < Q1 - 1.5*IQR) |
                (self.df[col] > Q3 + 1.5*IQR))
        # On flag uniquement si plus de 1% d'outliers
        if mask.sum() / len(self.df) > 0.01:
            self.df[f'{col}_outlier'] = mask.astype(int)
            n_outliers += mask.sum()
    print(f'   T6 Outliers flagués          : {n_outliers}')
    self.rapport.append(f'T6 Outliers : {n_outliers} flagués')

    # T7 — Encodage ML
    le       = LabelEncoder()
    cat_cols = self.df.select_dtypes(include='object').columns
    n_encoded = 0
    for col in cat_cols:
        if self.df[col].nunique() < 50:
            self.df[f'{col}_encoded'] = le.fit_transform(
                self.df[col].astype(str)
            )
            n_encoded += 1
    print(f'   T7 Colonnes encodees         : {n_encoded}')
    self.rapport.append(f'T7 Encodage : {n_encoded} colonnes')

    # T8 — Score completude
    quality_cols = [c for c in self.df.select_dtypes(include='object').columns
                    if '_encoded' not in c and 'name' not in c
                    and 'week' not in c][:5]
    self.df['completeness_score'] = (
        self.df[quality_cols].apply(lambda row: sum(
            1 for v in row
            if pd.notna(v) and str(v) not in ['Unknown','nan','']
        ), axis=1) / len(quality_cols) * 100
    ).round(1)
    print(f'   T8 Score completude calcule')
    self.rapport.append('T8 Score completude calcule')
    return self
# ══════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", [
        "Upload Dataset",
        "Audit Qualite",
        "ETL Transformation",
        "Visualisations",
        "Modele ML",
        "Rapport PDF"
    ])
    st.markdown("---")
    st.markdown("**ETL Automatique v1.0**")
    st.markdown("Projet IA/ML — Anaconda")

# ── Session state ──────────────────────────────────────────────────────
if 'etl' not in st.session_state:
    st.session_state.etl = None
if 'transformed' not in st.session_state:
    st.session_state.transformed = False

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════
if page == "Upload Dataset":
    st.header("Upload ton Dataset")

    uploaded = st.file_uploader(
        "Glisse ton fichier ici",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded:
        try:
            ext = uploaded.name.split('.')[-1].lower()
            if ext == 'csv':
                sample = uploaded.read(2000).decode('utf-8', errors='ignore')
                uploaded.seek(0)
                sep = ';' if sample.count(';') > sample.count(',') else ','
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(uploaded, sep=sep, encoding=enc,
                                         on_bad_lines='skip')
                        uploaded.seek(0)
                        break
                    except:
                        uploaded.seek(0)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')

            st.session_state.etl = ETLAuto(df, filename=uploaded.name)
            st.session_state.transformed = False

            st.success(f"Dataset charge avec succes !")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes",    f"{df.shape[0]:,}")
            col2.metric("Colonnes",  f"{df.shape[1]}")
            col3.metric("Manquants", f"{df.isnull().sum().sum():,}")
            col4.metric("Doublons",  f"{df.duplicated().sum():,}")

            st.subheader("Apercu des donnees")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.info("Uploade un fichier CSV ou Excel pour commencer !")

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — AUDIT
# ══════════════════════════════════════════════════════════════════════
elif page == "Audit Qualite":
    st.header("Audit Qualite des Donnees")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    else:
        etl = st.session_state.etl
        audit_df = etl.audit()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rapport des valeurs manquantes")
            st.dataframe(audit_df, use_container_width=True)

        with col2:
            st.subheader("Visualisation")
            missing = etl.df_raw.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(missing.index, missing.values, color='#E50914')
                ax.set_xlabel('Nombre de valeurs manquantes')
                ax.set_title('Valeurs manquantes par colonne')
                for bar, val in zip(ax.patches, missing.values):
                    ax.text(bar.get_width() + 1,
                            bar.get_y() + bar.get_height()/2,
                            f'{val}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.success("Aucune valeur manquante !")

        st.subheader("Statistiques generales")
        st.dataframe(etl.df_raw.describe(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════
elif page == "ETL Transformation":
    st.header("ETL — Transformations Automatiques")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    else:
        etl = st.session_state.etl

        if not st.session_state.transformed:
            st.info("Clique sur le bouton pour lancer toutes les transformations automatiquement !")
            if st.button("Lancer le Pipeline ETL", type="primary", use_container_width=True):
                with st.spinner("Transformation en cours..."):
                    log = etl.transform()
                    st.session_state.transformed = True
                st.success("Pipeline ETL termine !")
                for item in log:
                    st.write(f"✅ {item}")
        else:
            st.success("Pipeline ETL deja execute !")
            for item in etl.rapport:
                st.write(f"✅ {item}")

        if st.session_state.transformed:
            col1, col2, col3 = st.columns(3)
            col1.metric("Lignes originales", f"{len(etl.df_raw):,}")
            col2.metric("Lignes finales",    f"{len(etl.df):,}")
            col3.metric("Score completude",  f"{etl.df['completeness_score'].mean():.1f}%")

            st.subheader("Dataset transforme")
            st.dataframe(etl.df.head(10), use_container_width=True)

            col_csv, col_excel = st.columns(2)
            with col_csv:
                csv = etl.df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "Telecharger CSV",
                    data=csv,
                    file_name=f'{etl.filename}_ETL.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            with col_excel:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    etl.df.to_excel(w, index=False, sheet_name='Data_Cleaned')
                buf.seek(0)
                st.download_button(
                    "Telecharger Excel",
                    data=buf,
                    file_name=f'{etl.filename}_ETL.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════
elif page == "Visualisations":
    st.header("Visualisations Automatiques")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl = st.session_state.etl
        df  = etl.df

        num_cols = [c for c in df.select_dtypes(include=np.number).columns
                    if '_outlier' not in c and '_encoded' not in c
                    and c != 'completeness_score']
        cat_cols = [c for c in df.select_dtypes(include='object').columns
                    if df[c].nunique() < 20]

        if num_cols:
            st.subheader("Distribution colonnes numeriques")
            n = min(len(num_cols), 3)
            cols = st.columns(n)
            for i, col in enumerate(num_cols[:n]):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(df[col].dropna(), bins=30,
                            color='#3498DB', edgecolor='white')
                    ax.axvline(df[col].mean(), color='red',
                               linestyle='--', label=f'Moy: {df[col].mean():.1f}')
                    ax.set_title(col, fontsize=10)
                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        if cat_cols:
            st.subheader("Distribution colonnes categorielles")
            n = min(len(cat_cols), 3)
            cols = st.columns(n)
            for i, col in enumerate(cat_cols[:n]):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    top = df[col].value_counts().head(8)
                    ax.barh(top.index[::-1], top.values[::-1], color='#9B59B6')
                    ax.set_title(col, fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        if len(num_cols) >= 2:
            st.subheader("Matrice de correlation")
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f',
                        cmap='coolwarm', ax=ax, linewidths=0.5)
            ax.set_title('Matrice de Correlation')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — ML
# ══════════════════════════════════════════════════════════════════════
elif page == "Modele ML":
    st.header("Modele ML Automatique")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl = st.session_state.etl
        num_cols = [c for c in etl.df.select_dtypes(include=np.number).columns
                    if '_outlier' not in c and '_encoded' not in c
                    and c != 'completeness_score']

        if len(num_cols) < 2:
            st.error("Pas assez de colonnes numeriques pour le ML !")
        else:
            target = st.selectbox(
                "Choisis la colonne cible (ce que tu veux predire)",
                num_cols
            )

            if st.button("Entrainer le Modele ML", type="primary", use_container_width=True):
                with st.spinner("Entrainement en cours..."):
                    model, score, importances, prob_type = etl.auto_ml(target)

                if model is None:
                    st.error("Erreur lors de l'entrainement !")
                else:
                    st.success(f"Modele entraine ! Type : {prob_type.upper()}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Resultats")
                        if prob_type == 'classification':
                            st.metric("Accuracy", f"{score['accuracy']:.2f}%")
                        else:
                            st.metric("R2 Score", f"{score['r2']:.2f}%")
                            st.metric("MAE",      f"{score['mae']:.2f}")

                    with col2:
                        st.subheader("Importance des features")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.barh(importances['Feature'][::-1],
                                importances['Importance'][::-1],
                                color='#E74C3C')
                        ax.set_title(f'Prediction : {target}')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    st.dataframe(importances, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — RAPPORT PDF
# ══════════════════════════════════════════════════════════════════════
elif page == "Rapport PDF":
    st.header("Rapport PDF Automatique")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl = st.session_state.etl
        st.info("Clique sur le bouton pour generer et telecharger ton rapport PDF !")

        if st.button("Generer le Rapport PDF", type="primary", use_container_width=True):
            with st.spinner("Generation du PDF..."):
                pdf_buf = etl.generer_pdf()
            st.success("Rapport PDF genere !")
            st.download_button(
                "Telecharger le Rapport PDF",
                data=pdf_buf,
                file_name=f'Rapport_ETL_{etl.filename}.pdf',
                mime='application/pdf',
                use_container_width=True
            )
