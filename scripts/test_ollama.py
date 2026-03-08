#!/usr/bin/env python3
"""
test_ollama.py — Vérifie que Ollama est installé et fonctionne localement.

Usage:
    python scripts/test_ollama.py
    python scripts/test_ollama.py --model qwen3:8b
    python scripts/test_ollama.py --url http://localhost:11434

Dépendances: stdlib uniquement (aucun pip requis)
"""

import sys
import json
import time
import argparse
import urllib.request
import urllib.error

# ─── Configuration par défaut ────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_URL   = "http://localhost:11434"
TIMEOUT_S     = 60  # secondes

# ─── Helpers ─────────────────────────────────────────────────────────────────

def separator(char="─", width=60):
    print(char * width)

def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def err(msg):  print(f"  ❌  {msg}")
def info(msg): print(f"  ℹ️   {msg}")


def get_json(url: str) -> dict | None:
    """GET JSON depuis une URL. Retourne None en cas d'erreur."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError:
        return None
    except json.JSONDecodeError:
        return None


def post_json(url: str, payload: dict) -> dict | None:
    """POST JSON et retourne la réponse parsée. Retourne None en cas d'erreur."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        raise ConnectionError(str(e)) from e
    except json.JSONDecodeError as e:
        raise ValueError("Réponse JSON invalide") from e


# ─── Étapes de test ──────────────────────────────────────────────────────────

def step_ping(base_url: str) -> bool:
    """Étape 1 : Ollama répond-il sur le port 11434 ?"""
    separator()
    print("Étape 1 — Connexion au serveur Ollama")
    separator()

    data = get_json(f"{base_url}/api/tags")
    if data is None:
        err("Impossible de contacter Ollama.")
        print()
        warn("Causes possibles :")
        info("  → Ollama n'est pas démarré  : lancez 'ollama serve' dans un terminal")
        info("  → Pare-feu Windows bloque le port 11434")
        info("  → URL incorrecte (défaut : http://localhost:11434)")
        return False

    ok(f"Serveur Ollama joignable → {base_url}")
    return True


def step_list_models(base_url: str) -> list[str]:
    """Étape 2 : Quels modèles sont disponibles ?"""
    separator()
    print("Étape 2 — Modèles disponibles localement")
    separator()

    data = get_json(f"{base_url}/api/tags") or {}
    models = [m["name"] for m in data.get("models", [])]

    if not models:
        warn("Aucun modèle installé.")
        info("  → Installez un modèle avec : ollama pull qwen3:8b")
    else:
        ok(f"{len(models)} modèle(s) trouvé(s) :")
        for m in models:
            print(f"       • {m}")
    return models


def step_inference(base_url: str, model: str) -> bool:
    """Étape 3 : Test d'inférence avec le modèle choisi."""
    separator()
    print(f"Étape 3 — Test d'inférence  (modèle : {model})")
    separator()

    prompt = (
        "Réponds en une seule phrase courte en français : "
        "quel est le résultat de 12 × 8 ?"
    )
    info(f"Prompt : {prompt}")
    print()

    try:
        t0 = time.perf_counter()
        resp = post_json(f"{base_url}/api/generate", {
            "model":  model,
            "prompt": prompt,
            "stream": False,
        })
        elapsed = time.perf_counter() - t0
    except ConnectionError as e:
        err(f"Erreur de connexion pendant l'inférence : {e}")
        return False
    except ValueError as e:
        err(f"Réponse invalide : {e}")
        return False

    if resp is None or "response" not in resp:
        err("Réponse vide ou format inattendu.")
        info(f"Réponse brute : {resp}")
        return False

    text = resp["response"].strip()
    tokens  = resp.get("eval_count", 0)
    t_load  = resp.get("load_duration",  0) / 1e9   # ns → s
    t_total = resp.get("total_duration", 0) / 1e9

    ok(f"Réponse reçue en {elapsed:.2f}s")
    print()
    print(f'  📝  "{text}"')
    print()

    if tokens and elapsed > 0:
        tps = tokens / elapsed
        info(f"Tokens générés : {tokens}  |  Débit : {tps:.1f} t/s")
    if t_load > 0:
        info(f"Chargement modèle : {t_load:.1f}s  |  Total Ollama : {t_total:.1f}s")

    return True


def step_api_compat(base_url: str, model: str) -> bool:
    """Étape 4 : Test compatibilité API OpenAI (/v1/chat/completions)."""
    separator()
    print("Étape 4 — Compatibilité API OpenAI (optionnel)")
    separator()

    try:
        resp = post_json(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Dis juste : OK"}],
        })
    except (ConnectionError, ValueError):
        warn("Endpoint /v1/chat/completions non disponible.")
        info("  → Normal sur certaines versions d'Ollama antérieures à 0.2")
        return False

    if resp and resp.get("choices"):
        msg = resp["choices"][0].get("message", {}).get("content", "").strip()
        ok(f"API OpenAI compatible  →  réponse : \"{msg}\"")
        info("  → Ouroboros pourra utiliser ce endpoint directement")
        return True

    warn("Réponse /v1 inattendue — compatibilité partielle.")
    return False


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Teste une installation Ollama locale."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modèle à tester (défaut : {DEFAULT_MODEL})")
    parser.add_argument("--url",   default=DEFAULT_URL,
                        help=f"URL du serveur Ollama (défaut : {DEFAULT_URL})")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    model    = args.model

    separator("═")
    print("  🔍  Test Ollama — vérification de l'installation locale")
    separator("═")
    print(f"  URL    : {base_url}")
    print(f"  Modèle : {model}")
    separator("═")
    print()

    # Étape 1 — ping
    if not step_ping(base_url):
        separator("═")
        err("Test interrompu : serveur inaccessible.")
        separator("═")
        sys.exit(1)

    # Étape 2 — liste modèles
    available = step_list_models(base_url)

    # Vérifier que le modèle demandé est présent
    if model not in available:
        # Cherche une correspondance partielle (ex. "qwen3" dans "qwen3:8b")
        matches = [m for m in available if model.split(":")[0] in m]
        if matches:
            model = matches[0]
            warn(f"Modèle exact non trouvé — utilisation de : {model}")
        else:
            separator()
            err(f"Modèle '{model}' non disponible localement.")
            info(f"  → Pour l'installer : ollama pull {model}")
            if available:
                info(f"  → Modèles disponibles : {', '.join(available)}")
            separator("═")
            err("Test interrompu : modèle manquant.")
            separator("═")
            sys.exit(1)

    # Étape 3 — inférence
    inference_ok = step_inference(base_url, model)

    # Étape 4 — compat OpenAI (non bloquant)
    compat_ok = False
    if inference_ok:
        compat_ok = step_api_compat(base_url, model)

    # ─── Résumé ─────────────────────────────────────────────────────────────
    separator("═")
    print("  📋  Résumé")
    separator("═")
    ok("Serveur Ollama : en ligne")
    (ok if inference_ok else err)(f"Inférence ({model}) : {'OK' if inference_ok else 'ÉCHEC'}")
    (ok if compat_ok  else warn)(f"API OpenAI compat   : {'OK' if compat_ok else 'non disponible'}")
    separator("═")

    if inference_ok:
        print()
        ok("Ollama est opérationnel sur ta machine.")
        print()
        info("Prochaine étape — connecter Ouroboros à ce modèle local :")
        info("  Configurer OPENROUTER_BASE_URL=http://localhost:11434/v1")
        info("  et OUROBOROS_MODEL=qwen3:8b dans l'environnement Colab")
        print()
    else:
        print()
        err("Ollama ne répond pas correctement. Consulte les messages ci-dessus.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
