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

st.set_page_config(page_title="ETL Smart", page_icon="ETL", layout="wide")

st.markdown("""
<style>
    .main-title { font-size:2.5rem; font-weight:bold; color:#E50914; text-align:center; margin-bottom:0.5rem; }
    .sub-title  { text-align:center; color:#666; margin-bottom:2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Pipeline ETL Smart</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload n\'importe quel dataset — le pipeline s\'adapte automatiquement</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CLASSE ETL SMART V2
# ══════════════════════════════════════════════════════════════════════
class ETLSmart:

    def __init__(self, df, filename='dataset'):
        self.df_raw    = df.copy()
        self.df        = df.copy()
        self.filename  = filename
        self.rapport   = []
        self.id_cols   = []
        self.date_cols = []
        self.num_cols  = []
        self.cat_cols  = []

    def _detecter_types(self):
        id_keywords = ['id','_id','code','ref','num','no','number','key']
        self.id_cols = [
            c for c in self.df.columns
            if any(k == c.lower() or c.lower().startswith(k+'_')
                   or c.lower().endswith('_'+k) for k in id_keywords)
        ]
        date_keywords = ['date','time','created','updated','born','start','end','added','at']
        potential_dates = [
            c for c in self.df.select_dtypes(include='object').columns
            if any(k in c.lower() for k in date_keywords)
        ]
        self.date_cols = []
        for col in potential_dates:
            try:
                parsed = pd.to_datetime(self.df[col].dropna().head(20), dayfirst=True, errors='coerce')
                if parsed.notna().sum() >= 5:
                    self.date_cols.append(col)
            except:
                pass
        self.num_cols = [c for c in self.df.select_dtypes(include=np.number).columns if c not in self.id_cols]
        self.cat_cols = [c for c in self.df.select_dtypes(include='object').columns
                         if c not in self.id_cols and c not in self.date_cols]

    def audit(self):
        self._detecter_types()
        df_audit    = self.df.drop(columns=self.id_cols, errors='ignore')
        missing     = df_audit.isnull().sum()
        missing_pct = (missing / len(df_audit) * 100).round(2)
        return pd.DataFrame({
            'Manquants': missing, 'Pourcentage %': missing_pct, 'Type': df_audit.dtypes
        }).sort_values('Manquants', ascending=False)

    def transform(self):
        log = []
        self._detecter_types()

        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        log.append(f'T1 - Doublons supprimes : {before - len(self.df)}')

        cols_to_clean = [c for c in self.df.select_dtypes(include='object').columns
                         if c not in self.id_cols and c not in self.date_cols]
        for col in cols_to_clean:
            self.df[col] = self.df[col].astype(str).str.strip().str.title()
        log.append(f'T2 - Espaces et casse normalises : {len(cols_to_clean)} colonnes')

        n_total = 0
        for col in self.df.columns:
            if col in self.id_cols:
                continue
            n_miss  = self.df[col].isin(['nan','NaN','None','']).sum()
            n_miss += self.df[col].isnull().sum()
            if n_miss > 0:
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].replace(
                        {'nan':'Unknown','Nan':'Unknown','None':'Unknown','':'Unknown'}
                    ).fillna('Unknown')
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                n_total += n_miss
        log.append(f'T3 - Imputation : {n_total} valeurs')

        n_dates = 0
        for col in self.date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], dayfirst=True, errors='coerce')
                prefix = col.lower().replace('date','').replace('time','').strip('_') or col.lower()
                self.df[f'year_{prefix}']       = self.df[col].dt.year.astype('Int64')
                self.df[f'month_{prefix}']      = self.df[col].dt.month.astype('Int64')
                self.df[f'day_{prefix}']        = self.df[col].dt.day.astype('Int64')
                self.df[f'quarter_{prefix}']    = self.df[col].dt.quarter.astype('Int64')
                self.df[f'month_name_{prefix}'] = self.df[col].dt.strftime('%B')
                self.df[f'dayofweek_{prefix}']  = self.df[col].dt.strftime('%A')
                self.df[f'is_weekend_{prefix}'] = self.df[col].dt.dayofweek.isin([5,6]).astype(int)
                n_dates += 1
            except:
                pass
        log.append(f'T4 - Dates : {n_dates} colonnes converties (7 features chacune)')

        n_errors = self._verifier_coherence()
        log.append(f'T5 - Coherence : {n_errors} erreurs corrigees')

        n_cat = self._categoriser_numeriques()
        log.append(f'T6 - Categorisation : {n_cat} colonnes')

        n_out = 0
        for col in self.num_cols:
            if col not in self.df.columns:
                continue
            Q1   = self.df[col].quantile(0.25)
            Q3   = self.df[col].quantile(0.75)
            IQR  = Q3 - Q1
            mask = ((self.df[col] < Q1-1.5*IQR) | (self.df[col] > Q3+1.5*IQR))
            if mask.sum() / len(self.df) > 0.05:
                self.df[f'{col}_outlier'] = mask.astype(int)
                n_out += mask.sum()
        log.append(f'T7 - Outliers : {n_out} flagués')

        le = LabelEncoder()
        n_encoded = 0
        cols_to_encode = [
            c for c in self.df.select_dtypes(include='object').columns
            if c not in self.id_cols and c not in self.date_cols
            and 'name' not in c.lower() and 'dayofweek' not in c.lower()
            and 'description' not in c.lower() and 'title' not in c.lower()
            and 'cast' not in c.lower() and self.df[c].nunique() < 50
        ]
        for col in cols_to_encode:
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            n_encoded += 1
        log.append(f'T8 - Encodage ML : {n_encoded} colonnes')

        quality_cols = [c for c in self.cat_cols if c in self.df.columns][:6]
        if quality_cols:
            self.df['completeness_score'] = (
                self.df[quality_cols].apply(lambda row: sum(
                    1 for v in row if pd.notna(v) and str(v) not in ['Unknown','Nan','']
                ), axis=1) / len(quality_cols) * 100
            ).round(1)
        else:
            self.df['completeness_score'] = 100.0
        log.append('T9 - Score completude calcule')

        self.rapport = log
        return log

    def _verifier_coherence(self):
        num_cols   = [c for c in self.num_cols if c in self.df.columns]
        n_errors   = 0
        price_kw   = ['price','prix','cost','rate','tarif','unit']
        qty_kw     = ['qty','quantity','units','sold','qte','nb','count']
        total_kw   = ['total','revenue','sales','amount','sum','turnover']
        price_cols = [c for c in num_cols if any(k in c.lower() for k in price_kw)]
        qty_cols   = [c for c in num_cols if any(k in c.lower() for k in qty_kw)]
        total_cols = [c for c in num_cols if any(k in c.lower() for k in total_kw)]
        for p in price_cols:
            for q in qty_cols:
                for t in total_cols:
                    if p == q or p == t or q == t:
                        continue
                    expected = (self.df[p] * self.df[q]).round(2)
                    errors   = abs(self.df[t] - expected) > 1
                    if errors.sum() > 0 and errors.sum() < len(self.df) * 0.1:
                        self.df.loc[errors, t] = expected[errors]
                        n_errors += errors.sum()
        return n_errors

    def _categoriser_numeriques(self):
        n_cat = 0
        for col in self.num_cols:
            if col not in self.df.columns:
                continue
            cv = self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else 0
            if cv > 0.3 and self.df[col].nunique() > 10:
                Q1 = self.df[col].quantile(0.33)
                Q3 = self.df[col].quantile(0.66)
                labels = ['Budget','Mid-Range','Premium'] if any(
                    k in col.lower() for k in ['price','prix','cost','revenue','sales']
                ) else ['Low','Medium','High']
                self.df[f'{col}_category'] = pd.cut(
                    self.df[col],
                    bins=[self.df[col].min()-1, Q1, Q3, self.df[col].max()+1],
                    labels=labels
                ).astype(str)
                n_cat += 1
        return n_cat

    def auto_ml(self, target):
        feature_cols = [
            c for c in self.df.select_dtypes(include=np.number).columns
            if c not in self.id_cols and '_outlier' not in c
            and c != 'completeness_score' and c != target
        ]
        if len(feature_cols) == 0:
            return None, None, None, None
        X = self.df[feature_cols].fillna(0)
        y = self.df[target].fillna(0)
        prob_type = 'classification' if y.nunique() <= 10 else 'regression'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42) if prob_type == 'classification' \
                else RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importances = pd.DataFrame({
            'Feature': feature_cols, 'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        score = {'type': prob_type,
                 'accuracy': accuracy_score(y_test, y_pred)*100} if prob_type == 'classification' \
                else {'type': prob_type,
                      'mae': mean_absolute_error(y_test, y_pred),
                      'r2': r2_score(y_test, y_pred)*100}
        return model, score, importances, prob_type

    def generer_pdf(self):
        def clean(text):
            return str(text).encode('latin-1', errors='ignore').decode('latin-1')
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_fill_color(229, 9, 20)
        pdf.rect(0, 0, 210, 35, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_xy(10, 8)
        pdf.cell(0, 10, 'RAPPORT ETL SMART v2', ln=True, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.set_xy(10, 22)
        pdf.cell(0, 8, f'Genere le : {datetime.datetime.now().strftime("%d/%m/%Y a %H:%M")}', ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(10, 45)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, '  1. INFORMATIONS DATASET', ln=True, fill=True)
        pdf.ln(3)
        for label, val in [
            ('Fichier', clean(self.filename)),
            ('Lignes originales', f'{len(self.df_raw):,}'),
            ('Lignes finales', f'{len(self.df):,}'),
            ('Colonnes originales', f'{self.df_raw.shape[1]}'),
            ('Colonnes finales', f'{self.df.shape[1]}'),
            ('Colonnes ID', clean(str(self.id_cols))),
            ('Colonnes Dates', clean(str(self.date_cols))),
            ('Score completude', f'{self.df["completeness_score"].mean():.1f}%'),
        ]:
            pdf.set_x(15)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(80, 7, clean(label))
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(100, 7, clean(val), ln=True)
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, '  2. TRANSFORMATIONS APPLIQUEES', ln=True, fill=True)
        pdf.ln(3)
        pdf.set_font('Helvetica', '', 10)
        for i, item in enumerate(self.rapport):
            pdf.set_x(15)
            pdf.cell(190, 7, clean(f'{i+1}. {item}'), ln=True)
        pdf.ln(4)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(190, 10, '  3. AUDIT QUALITE', ln=True, fill=True)
        pdf.ln(3)
        pdf.set_fill_color(52, 73, 94)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_x(15)
        pdf.cell(70, 7, 'Colonne', fill=True, border=1)
        pdf.cell(35, 7, 'Type', fill=True, border=1)
        pdf.cell(40, 7, 'Manquants', fill=True, border=1)
        pdf.cell(40, 7, 'Pct %', fill=True, border=1, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 8)
        missing     = self.df_raw.isnull().sum()
        missing_pct = (missing / len(self.df_raw) * 100).round(2)
        for i, col in enumerate(self.df_raw.columns):
            fill = i % 2 == 0
            pdf.set_fill_color(245,245,245) if fill else pdf.set_fill_color(255,255,255)
            pdf.set_x(15)
            pdf.cell(70, 6, clean(col[:28]), fill=fill, border=1)
            pdf.cell(35, 6, clean(str(self.df_raw[col].dtype)), fill=fill, border=1)
            pdf.cell(40, 6, clean(str(missing[col])), fill=fill, border=1)
            pdf.cell(40, 6, clean(f'{missing_pct[col]:.2f}%'), fill=fill, border=1, ln=True)
        pdf.ln(8)
        pdf.set_fill_color(229, 9, 20)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(190, 9, '  Pipeline ETL Smart v2 - Python & Streamlit', fill=True, ln=True, align='C')
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", [
        "Upload Dataset", "Audit Qualite", "ETL Transformation",
        "Visualisations", "Modele ML", "Rapport PDF"
    ])
    st.markdown("---")
    st.markdown("**ETL Smart v2.0**")
    st.markdown("Pipeline Auto-Adaptatif")
    st.markdown("Projet IA/ML - Anaconda")

if 'etl' not in st.session_state:
    st.session_state.etl = None
if 'transformed' not in st.session_state:
    st.session_state.transformed = False

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════
if page == "Upload Dataset":
    st.header("Upload ton Dataset")
    st.info("Le pipeline detecte automatiquement : separateur, encodage, dates, ID, coherence, outliers, categories...")

    uploaded = st.file_uploader("Glisse ton fichier ici (CSV ou Excel)", type=['csv','xlsx','xls'])

    if uploaded:
        try:
            ext = uploaded.name.split('.')[-1].lower()
            if ext == 'csv':
                sample = uploaded.read(2000).decode('utf-8', errors='ignore')
                uploaded.seek(0)
                sep = ';' if sample.count(';') > sample.count(',') else ','
                df  = None
                for enc in ['utf-8','latin-1','cp1252']:
                    try:
                        df = pd.read_csv(uploaded, sep=sep, encoding=enc, on_bad_lines='skip')
                        uploaded.seek(0)
                        break
                    except:
                        uploaded.seek(0)
            else:
                df = pd.read_excel(uploaded, engine='openpyxl')

            st.session_state.etl = ETLSmart(df, filename=uploaded.name)
            st.session_state.etl._detecter_types()
            st.session_state.transformed = False
            st.success("Dataset charge avec succes !")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes",    f"{df.shape[0]:,}")
            col2.metric("Colonnes",  f"{df.shape[1]}")
            col3.metric("Manquants", f"{df.isnull().sum().sum():,}")
            col4.metric("Doublons",  f"{df.duplicated().sum():,}")

            etl = st.session_state.etl
            col_a, col_b = st.columns(2)
            with col_a:
                st.success(f"ID detectes : {etl.id_cols if etl.id_cols else 'aucun'}")
                st.success(f"Dates detectees : {etl.date_cols if etl.date_cols else 'aucune'}")
            with col_b:
                st.info(f"Colonnes numeriques : {len(etl.num_cols)}")
                st.info(f"Colonnes categorielles : {len(etl.cat_cols)}")

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
        etl      = st.session_state.etl
        audit_df = etl.audit()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rapport valeurs manquantes")
            st.dataframe(audit_df, use_container_width=True)
        with col2:
            st.subheader("Visualisation")
            missing = etl.df_raw.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(missing.index, missing.values, color='#E50914')
                ax.set_xlabel('Valeurs manquantes')
                ax.set_title('Valeurs manquantes par colonne')
                for bar, val in zip(ax.patches, missing.values):
                    ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                            f'{val}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.success("Aucune valeur manquante !")
        st.subheader("Statistiques generales")
        df_stats = etl.df_raw.drop(columns=etl.id_cols, errors='ignore')
        st.dataframe(df_stats.describe(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════
elif page == "ETL Transformation":
    st.header("ETL Smart - Transformations Automatiques")
    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    else:
        etl = st.session_state.etl
        if not st.session_state.transformed:
            st.info("Le pipeline s'adapte automatiquement a la structure de ton dataset !")
            if st.button("Lancer le Pipeline ETL Smart", type="primary", use_container_width=True):
                with st.spinner("Pipeline en cours..."):
                    log = etl.transform()
                    st.session_state.transformed = True
                st.success("Pipeline ETL Smart termine !")
                for item in log:
                    st.write(f"OK {item}")
        else:
            st.success("Pipeline ETL deja execute !")
            for item in etl.rapport:
                st.write(f"OK {item}")

        if st.session_state.transformed:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes originales", f"{len(etl.df_raw):,}")
            col2.metric("Lignes finales",    f"{len(etl.df):,}")
            col3.metric("Colonnes finales",  f"{etl.df.shape[1]}")
            col4.metric("Score completude",  f"{etl.df['completeness_score'].mean():.1f}%")
            st.subheader("Dataset transforme")
            st.dataframe(etl.df.head(10), use_container_width=True)
            col_csv, col_excel = st.columns(2)
            with col_csv:
                csv = etl.df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("Telecharger CSV", data=csv,
                    file_name=f'{etl.filename}_ETL_Smart.csv', mime='text/csv',
                    use_container_width=True)
            with col_excel:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                    etl.df.to_excel(w, index=False, sheet_name='Data_Cleaned')
                buf.seek(0)
                st.download_button("Telecharger Excel", data=buf,
                    file_name=f'{etl.filename}_ETL_Smart.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True)

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
                    and c != 'completeness_score' and c not in etl.id_cols]
        cat_cols = [c for c in df.select_dtypes(include='object').columns
                    if df[c].nunique() < 20 and 'dayofweek' not in c.lower()
                    and 'month_name' not in c.lower()]
        if num_cols:
            st.subheader("Distribution colonnes numeriques")
            n = min(len(num_cols), 3)
            cols = st.columns(n)
            for i, col in enumerate(num_cols[:n]):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(df[col].dropna(), bins=30, color='#3498DB', edgecolor='white')
                    ax.axvline(df[col].mean(), color='red', linestyle='--',
                               label=f'Moy: {df[col].mean():.1f}')
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
            sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f',
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
        etl      = st.session_state.etl
        num_cols = [c for c in etl.df.select_dtypes(include=np.number).columns
                    if '_outlier' not in c and '_encoded' not in c
                    and c != 'completeness_score' and c not in etl.id_cols]
        if len(num_cols) < 2:
            st.error("Pas assez de colonnes numeriques pour le ML !")
        else:
            st.info("Classification ou Regression detecte automatiquement selon la colonne cible")
            target = st.selectbox("Choisis la colonne cible", num_cols)
            if st.button("Entrainer le Modele ML", type="primary", use_container_width=True):
                with st.spinner("Entrainement en cours..."):
                    model, score, importances, prob_type = etl.auto_ml(target)
                if model is None:
                    st.error("Pas assez de features numeriques !")
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
                                importances['Importance'][::-1], color='#E74C3C')
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
        st.info("Le rapport inclut : dataset info, colonnes detectees, transformations, audit qualite")
        if st.button("Generer le Rapport PDF", type="primary", use_container_width=True):
            with st.spinner("Generation du PDF..."):
                pdf_buf = etl.generer_pdf()
            st.success("Rapport PDF genere !")
            st.download_button(
                "Telecharger le Rapport PDF", data=pdf_buf,
                file_name=f'Rapport_ETL_Smart_{etl.filename}.pdf',
                mime='application/pdf', use_container_width=True)
