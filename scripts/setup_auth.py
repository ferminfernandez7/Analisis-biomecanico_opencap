"""
Script de autenticación para OpenCap.
Ejecutar UNA VEZ para generar el archivo .env con tu token.

Uso:
    conda activate py311prue
    python scripts/setup_auth.py
"""
import requests
import getpass
import os

API_URL = "https://api.opencap.ai/"

print("=" * 60)
print("  Autenticación OpenCap")
print("  Usa tus credenciales de app.opencap.ai")
print("=" * 60)
print()

username = input("Usuario (email): ")
password = getpass.getpass("Contraseña: ")

try:
    resp = requests.post(
        API_URL + "login/",
        data={"username": username, "password": password}
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["token"]
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path, "w") as f:
        f.write(f'API_TOKEN="{token}"\n')
    
    print(f"\n✅ Login exitoso!")
    print(f"   Token guardado en: {env_path}")
    print(f"   NO subas este archivo a git.")
    
except Exception as e:
    print(f"\n❌ Error de autenticación: {e}")
    print("   Verifica tus credenciales en app.opencap.ai")
