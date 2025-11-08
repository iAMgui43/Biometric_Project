# Facial Access Prototype (OpenCV LBPH + Haar, sem Deep Learning)

Compatível com Python 3.13 no Windows sem TensorFlow/Keras/DeepFace.

## Como rodar
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Como funciona
- **Detecção de rosto:** Haar Cascade (OpenCV).
- **Reconhecimento:** LBPH (OpenCV `cv2.face` do pacote opencv-contrib-python).
- **Cadastro:** recorta o rosto detectado, converte para escala de cinza, redimensiona (200x200) e salva em `faces/<id>_<nome>.png`.
- **Treino:** a cada novo cadastro, o modelo LBPH é re-treinado com todas as amostras.
- **Verificação:** compara o frame atual com o modelo. Quanto **menor** o `confidence`, melhor o match (usa limiar 70).

## Estrutura
- `app.py` - rotas Flask
- `db.py` - usuários (id, nome, nível, caminho_da_imagem) + logs
- `face_utils.py` - detecção, treino e verificação LBPH
- `templates/` - UI com Tailwind
- `faces/` - imagens recortadas
- `lbph_model.yml` - modelo treinado (gerado após o primeiro cadastro)

## Observações
- Para melhor acurácia, cadastre 2–3 amostras por pessoa (refaça cadastro com nome igual).
- Ajuste `LBPH_THRESHOLD` em `face_utils.py` conforme sua câmera/ambiente (60–80).
