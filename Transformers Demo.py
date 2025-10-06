"""
# 🤖 Introdução aos Transformers e GPT

## Objetivo: Demonstrar geração de texto com modelos de linguagem
## Modelo: GPT-2 (OpenAI)
"""

from transformers import pipeline, set_seed
import matplotlib.pyplot as plt

def demo_transformers():
    # Carregar gerador de texto
    print("🚀 Carregando modelo GPT-2...")
    gerador = pipeline("text-generation", model="gpt2")
    set_seed(42)  # Para resultados reproduzíveis
    
    # Exemplos de prompts
    prompts = [
        "Qual a melhor época para visitar o Japão?",
        "Os benefícios de aprender machine learning são",
        "Como a inteligência artificial pode ajudar no dia a dia:"
    ]
    
    resultados = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*50}")
        print(f"📝 Prompt {i}: {prompt}")
        print(f"{'='*50}")
        
        # Gerar resposta
        resposta = gerador(
            prompt,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.2
        )
        
        texto_gerado = resposta[0]['generated_text']
        resultados.append({'prompt': prompt, 'resposta': texto_gerado})
        
        print(f"💬 Resposta: {texto_gerado}")
        
        # Análise simples
        palavras_prompt = len(prompt.split())
        palavras_resposta = len(texto_gerado.split())
        print(f"📊 Estatísticas: {palavras_prompt} → {palavras_resposta} palavras")
    
    return resultados

# Executar demonstração
print("🎯 Demonstração de Transformers e GPT-2")
print("=" * 60)
resultados = demo_transformers()

print(f"\n✅ Demonstração concluída! {len(resultados)} prompts processados.")