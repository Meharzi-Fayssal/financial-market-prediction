# /!\ A faire si duckdb n'est pas installé : pip install duckdb


import duckdb
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION
# ==========================================

debut_total = time.perf_counter()

repo = "hf://datasets/TnnnT0326/Jane_Street_Competition"

# Package 0 de chaque partition
url_0 = f"{repo}/train.parquet/partition_id=0/part-0.parquet"
url_1 = f"{repo}/train.parquet/partition_id=1/part-0.parquet"

# Dossier de sauvegarde
output_dir = Path(__file__).resolve().parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Connexion DuckDB
con = duckdb.connect()

print("[INFO] Connexion a DuckDB...")
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")

# Activer le calcul de progression DuckDB
con.execute("SET enable_progress_bar = true;")
con.execute("SET progress_bar_time = 3600000;")

print("[OK] DuckDB pret !")


# ==========================================
# 2. FONCTION DE SUIVI AVEC TQDM
# ==========================================

def executer_avec_progression(query, description):
    debut = time.perf_counter()

    def executer():
        return con.execute(query).fetchone()[0]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(executer)

        with tqdm(
            total=100,
            desc=description,
            unit="%",
            dynamic_ncols=True
        ) as pbar:

            while not future.done():

                # Progression estimee par DuckDB
                try:
                    progression = con.query_progress()

                    if progression is not None and progression >= 0:
                        valeur = min(99, int(progression))

                        if valeur > pbar.n:
                            pbar.update(valeur - pbar.n)

                except Exception:
                    pass

                time.sleep(0.3)

            # Recuperer le resultat ou propager l'erreur
            resultat = future.result()

            pbar.update(100 - pbar.n)

    duree = time.perf_counter() - debut

    print(f"[OK] {description} termine en {duree:.2f} s")

    return resultat


# ==========================================
# 3. VERIFICATION DES DATE_ID
# ==========================================

print("\n[INFO] Verification des plages de dates...")

def compter_jours(url, debut, fin, nom):
    query = f"""
        SELECT COUNT(DISTINCT date_id)
        FROM read_parquet('{url}', hive_partitioning=false)
        WHERE date_id BETWEEN {debut} AND {fin}
    """

    return executer_avec_progression(query, nom)


plages = [
    ("Train", url_0, 0, 109),
    ("Validation", url_0, 110, 153),
    ("Test partition 0", url_0, 154, 169),
    ("Test partition 1", url_1, 170, 219),
]

for nom, url, debut, fin in plages:
    attendu = fin - debut + 1
    trouve = compter_jours(url, debut, fin, nom)

    print(f"[CHECK] {nom}: {trouve}/{attendu} jours")

    if trouve != attendu:
        raise ValueError(
            f"Plage incomplete pour {nom}: "
            f"{trouve} jours trouves sur {attendu}"
        )

print("\n[OK] Toutes les plages sont valides !")


# ==========================================
# 4. SEGMENTATION
# ==========================================

train_query = f"""
    SELECT *
    FROM read_parquet('{url_0}', hive_partitioning=false)
    WHERE date_id BETWEEN 0 AND 109
"""

val_query = f"""
    SELECT *
    FROM read_parquet('{url_0}', hive_partitioning=false)
    WHERE date_id BETWEEN 110 AND 153
"""

test_query = f"""
    SELECT *
    FROM read_parquet('{url_0}', hive_partitioning=false)
    WHERE date_id BETWEEN 154 AND 169

    UNION ALL

    SELECT *
    FROM read_parquet('{url_1}', hive_partitioning=false)
    WHERE date_id BETWEEN 170 AND 219
"""


# ==========================================
# 5. SAUVEGARDE EN CSV AVEC PROGRESSION
# ==========================================

datasets = {
    "train": train_query,
    "val": val_query,
    "test": test_query
}

print("\n[INFO] Debut de la segmentation et des exports...")

for nom, query in datasets.items():

    fichier = (output_dir / f"{nom}.csv").as_posix()

    print(f"\n[INFO] Creation de {nom}.csv...")

    export_query = f"""
        COPY ({query})
        TO '{fichier}'
        (FORMAT CSV, HEADER TRUE)
    """

    nb_lignes = executer_avec_progression(
        export_query,
        f"Export {nom}"
    )

    taille_mo = Path(fichier).stat().st_size / (1024 ** 2)

    print(f"[OK] Fichier : {nom}.csv")
    print(f"[INFO] Nombre de lignes : {nb_lignes:,}")
    print(f"[INFO] Taille : {taille_mo:.2f} Mo")


# ==========================================
# 6. BILAN FINAL
# ==========================================

con.close()

duree_totale = time.perf_counter() - debut_total

minutes, secondes = divmod(duree_totale, 60)

print("\n" + "=" * 45)
print("SEGMENTATION TERMINEE")
print("=" * 45)

print(f"[OK] Train : {output_dir / 'train.csv'}")
print(f"[OK] Val   : {output_dir / 'val.csv'}")
print(f"[OK] Test  : {output_dir / 'test.csv'}")

print(
    f"\n[TEMPS TOTAL] "
    f"{int(minutes)} min {secondes:.2f} s"
)

print("=" * 45)
