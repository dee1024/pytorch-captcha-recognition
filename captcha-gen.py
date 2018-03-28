# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time

# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def random_captcha(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return ''.join(captcha_text)

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 2
    path = 'dataset/train'
    for i in range(count):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text+'_'+now+'.png'
        image.save(path +'/'+ filename)
        print('saved %d : %s' % (i+1,filename))

