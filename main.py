import discord
from discord.ext import commands
from model import get_class
from test import annotate_image

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)
result = 0
result2 = 0

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hello!! bot is ready, I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    # kode untuk bot menerima gambar
    if ctx.message.attachments: 
        for file in ctx.message.attachments: 
            file_name = file.filename 
            file_url = file.url
            await file.save(f'./{file.filename}')
            hasil = get_class(model_path='keras_model.h5', labels_path='labels.txt', image_path=f'./{file.filename}')
            
            if hasil[0] == 'Urban\n' and hasil[1] >= 0.65:
                await ctx.send('ini adalah Urban')
                await ctx.send('Pada tempat ini, seharusnya tidak banyak pohon dan memiliki tingkat polusi yang lebih tinggi')
                await ctx.send('(!) HANYA KLASIFIKASI (!) lakukan $combo untuk mengetahui lebih lanjut')
                result = 1
            else:
                await ctx.send('ini adalah Hutan')
                await ctx.send('Pada tempat ini, lebih banyak pohon, dan udara lebih segar, murni, minim polusi')
                await ctx.send('(!) HANYA KLASIFIKASI (!) lakukan $combo untuk mengetahui lebih lanjut')
                result = 2

            # kode untuk memproses gambar (ubah dengan melihat labels.txt)
            #if hasil[0] == 'Pohon\n' and hasil[1] >= 0.25:
                #await ctx.send('ini adalah Pohon')
                #await ctx.send('ini merupakan sampah anorganik, cukup berbahaya apabila dibiarkan')
                #await ctx.send('tetapi sampah ini bisa di daur ulang!!')
            #elif hasil[0] == 'Urban\n' and hasil[1] >= 0.65:
                #await ctx.send('ini adalah Urban')
                #await ctx.send('in merupakan sampah anorganik, cukup berbahaya apabila dibiarkan')
                #await ctx.send('tetapi sampah ini bisa di daur ulang')
            #else:
                #await ctx.send('GAMBAR MU KEMUNGKINAN: salah format/blur/corrupt')
                #await ctx.send('KIRIM GAMBAR BARU!!!')
    #else:
        #await ctx.send('GAMBAR TIDAK VALID/GAADA >:/')

@bot.command()
async def yolov5(ctx):
    if ctx.message.attachments:
        for file in ctx.message.attachments:
            # Save the uploaded image
            file_name = file.filename
            await file.save(f'./{file_name}')
            
            # Process the image with YOLOv5
            tra, pna = annotate_image(image_path=f'./{file_name}')
            
            # Respond with the results
            await ctx.send(f"telah dideteksi {tra} pohon dan {pna} kubangan air.")
            
            # Optionally send the annotated image
            await ctx.send(file=discord.File("output_image.jpg"))
    else:
        await ctx.send("No image uploaded.")


@bot.command()
async def combo(ctx):
    # Check if the user uploaded an image
    if ctx.message.attachments:
        for file in ctx.message.attachments:
            file_name = file.filename
            await file.save(f'./YoloV5/Images/{file.filename}')  # Save the uploaded file locally

            # Step 1: Perform classification
            hasil = get_class(model_path='keras_model.h5', labels_path='labels.txt', image_path=f'./YoloV5/Images/{file.filename}')
            
            # `hasil[0]` is the classification label, `hasil[1]` is the confidence score
            location_type = hasil[0].strip()  # Remove any trailing newline or whitespace
            confidence = hasil[1]
            
            # Step 2: Perform YOLOv5 detections
            tra, pna = annotate_image(image_path=f'./YoloV5/Images/{file_name}')
            
            # Send intermediate results
            await ctx.send(f"Lokasi terdeteksi sebagai **{location_type}**")
            await ctx.send(f"Terdeteksi **{tra} pohon** dan **{pna} kolam air**.")
            
            # Optionally send the annotated image
            await ctx.send(file=discord.File("output_image.jpg"))
            
            # Step 3: Apply combined logic for response
            if location_type == 'Urban' and confidence >= 0.65:
                if pna > 0:
                    await ctx.send("Ini adalah area urban dengan kolam air, kemungkinan terdapat taman atau area rekreasi!")
                elif tra > 2:
                    await ctx.send("Area urban ini memiliki banyak pohon, pohon pohon itu sepertinya bisa menyerap polusi area perkotaan!.")
                else:
                    await ctx.send("Area urban dengan sedikit pohon atau kolam, bisa jadi lingkungan perkotaan biasa.")
            
            else:
                if pna > 0:
                    await ctx.send("Udara segar, banyak pohon, ada kolam, pasti udara bersih dan minim polusi!")
                else:
                    await ctx.send("Hutan dengan banyak pohon, udara segar, dan suasana alami.")
           
    else:
        await ctx.send("Mohon unggah gambar untuk dianalisis.")



bot.run("TOKEN")