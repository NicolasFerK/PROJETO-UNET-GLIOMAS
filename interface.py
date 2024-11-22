import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, Scrollbar, Canvas
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
from rembg import remove

from tensorflow.keras.models import load_model

# carrega o modelo da ia
model = load_model('models/model2.h5')

# cria o vetor das imagens que escolhemos
imagens_selecionadas = []
# tamanho padrao das imagens a serem previstas pela ia
img_size = (128, 128)

# salva a pasta do aplicativo
pastaApp = os.path.dirname(__file__)


# classe principal
class GaleriaImagens:
    def __init__(self, master, canvas):
        self.master = master
        
        # titulo do app
        self.master.title("EL REMOVEDOR DE CEREBRO")
        
        self.canvas = canvas

        # lista para a imagem exibida e indice
        self.imagens = []
        self.index_imagem_atual = 0
        
        # label para a imagem a ser exibida
        self.label_imagem = Label(canvas, borderwidth=5, relief="solid")
        self.label_imagem.pack(pady=10, padx=10)

        # bind pra pular e voltar uma imagem
        root.bind("<Right>", self.proxima_imagem)
        root.bind("<Left>", self.anterior_imagem)

        # botões de pular e voltar uma imagem
        self.frame_botoes = Frame(canvas, highlightbackground="black", highlightthickness=5)
        self.frame_botoes.pack(side="bottom", pady=10)
        self.padxBtns = 50
        self.padyBtns = 10
        self.btn_anterior = Button(self.frame_botoes, text="Anterior", command=self.anterior_imagem)
        self.btn_anterior.pack(side="left", padx=self.padxBtns, pady=self.padyBtns)

        self.btn_selecionar = Button(self.frame_botoes, text="Selecionar Imagens", command=self.selecionar_imagens)
        self.btn_selecionar.pack(side="left", padx=self.padxBtns, pady=self.padyBtns)

        self.btn_proximo = Button(self.frame_botoes, text="Próximo", command=self.proxima_imagem)
        self.btn_proximo.pack(side="right", padx=self.padxBtns, pady=self.padyBtns)

    def selecionar_imagens(self):
        # abre tela de seleciona as imagens
        caminhos = filedialog.askopenfilenames(
            title="Selecionar imagens",
            filetypes=[("Arquivos de imagem", "*.jpg;*.jpeg;*.png;*.gif")]
        )
        
        self.imagens = []
        imagens_selecionadas.clear()  # limpa a lista antes de adiciona imagens

        for caminho in caminhos:
            print(caminho)
            
            # carrega e pre-processa as imagens
            image = tf.io.read_file(caminho)
            image = tf.image.decode_jpeg(image, channels=1)  # imagem em preto e branco
            image = tf.image.resize(image, img_size)
            image = image / 255.0  # normalizacao

            img_input = np.expand_dims(image, axis=0)
            img_input = np.expand_dims(img_input, axis=-1)

            # Predição do modelo
            pred_mask = model.predict(img_input)[0]  # remove a dimensao do batch
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # binariza

            # converte para imagem pil
            pil_mask = Image.fromarray(pred_mask.squeeze(), mode='L')

            cor_vermelho = (255, 0, 0) # cor vermelha definida

            imagem = pil_mask.convert('RGB') # converte a imagem pra rgb

            data = np.array(imagem) # pega as infos da imagem
            vermelho, verde, azul = data.T # anota cada valor do r, g e b

            # cria uma condição, para se os valores ultrapassarem aqueles a condicao = 1
            condicao = (vermelho >= 225) & (verde >= 225) & (azul >= 225)

            # quando condicao = 1 a cor do pixel vira vermelha
            data[condicao.T] = cor_vermelho

            # transforma de array pra imagem
            im2 = Image.fromarray(data)
            
            # remove fundo
            output = remove(im2)

            image_np = image.numpy()
            image_np = np.squeeze(image_np)
            # Verificar e ajustar o tipo de dados
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)

            # Criar a imagem PIL
            imagem_resized = Image.fromarray(image_np, mode='L').convert('RGB')

            output = output.convert('RGBA')

            mask = output.split()[3]

            output_rgb = output.convert('RGB')

            imagem_resized.paste(output_rgb, (0,0), mask = mask) 


            # transforma pra imagem tkinter
            photo = ImageTk.PhotoImage(imagem_resized)

            # salva no array das imagens
            self.imagens.append(photo)

            try:
                # Converter o tensor para NumPy
                image_np = image.numpy()

                # Remover a dimensão do canal se necessário
                image_np = np.squeeze(image_np)

                # Verificar e ajustar o tipo de dados
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)

                # Criar a imagem PIL
                imagem_resized = Image.fromarray(image_np, mode='L')  # 'L' para imagens em escala de cinza
                imagem_resized = imagem_resized.resize((100, 100))  # Redimensionar para 100x100

                # Converter para ImageTk.PhotoImage
                photo_small = ImageTk.PhotoImage(imagem_resized)
                imagens_selecionadas.append(photo_small)
            except Exception as e:
                print(f"Erro ao converter a imagem {caminho}: {e}")
        
        # Mostrar a primeira imagem após carregar tudo
        if self.imagens:
            self.index_imagem_atual = 0
            self.exibir_imagem()

    def exibir_imagem(self):
        # Atualiza a imagem principal
        self.label_imagem.config(image=self.imagens[self.index_imagem_atual])
        self.label_imagem.image = self.imagens[self.index_imagem_atual]
        
        # Limpar o frame de imagens selecionadas
        for widget in frame_imagens.winfo_children():
            widget.destroy()
        
        # Exibir todas as imagens selecionadas
        for img in imagens_selecionadas:
            label = Label(frame_imagens, image=img, borderwidth=5, relief="solid", width=100, height=100)
            label.pack(side="top", padx=10, pady=10)

    def proxima_imagem(self, event=None):
        if self.imagens:
            self.index_imagem_atual += 1
            if self.index_imagem_atual >= len(self.imagens):
                self.index_imagem_atual = 0  # Reseta se ultrapassar o máximo
            self.exibir_imagem()

    def anterior_imagem(self, event=None):
        if self.imagens:
            self.index_imagem_atual -= 1
            if self.index_imagem_atual < 0:
                self.index_imagem_atual = len(self.imagens) - 1  # Reseta se ultrapassar o mínimo
            self.exibir_imagem()

# cria o aplicativo 
root = tk.Tk()

# cria o frame da barrinha de rolar
frame_rolagem = Frame(root)
frame_rolagem.pack(side="right", fill="both", expand=True)

# cria o canvas que tem as img dentro
canvas = Canvas(frame_rolagem, bd=4, relief="solid")
canvas.pack(side="left", fill="both", expand=True)

# classe do app
galeria = GaleriaImagens(root, canvas)

# cria a barra de rolar
scrollbar = Scrollbar(frame_rolagem, orient="vertical", command=canvas.yview)
scrollbar.pack(side="left", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)

# cria o frame das img
frame_imagens = Frame(canvas, highlightbackground="black", highlightthickness=5)
canvas.create_window((25, 0), window=frame_imagens, anchor="nw")

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame_imagens.bind("<Configure>", on_configure)

root.mainloop()
