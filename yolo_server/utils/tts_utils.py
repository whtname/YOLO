#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tts_utils.py
# @Time      :2025/7/8 14:14:05
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :语音测试模块封装
# @Desc      :支持语音触发间隔,冷却间隔，以及语音触发
import pyttsx3
import time
import logging
logger = logging.getLogger(__name__)

def init_tts() -> pyttsx3.Engine | None:
    """
    初始化语音合成引擎
    :return: 返回语音合成引擎
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # 设置语速
        engine.setProperty('volume', 1.0)  # 设置音量
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or 'chinese' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        logger.info(f"初始化语音合成引擎失败: {e}")
        return None

def process_tts_detection(result, tts_enable, tts_duration,
                        tts_interval,tts_engine,tts_state,tts_text):
    """
    处理检测结果，并触发语音
    :param result: 检测结果
    :param tts_enable: 是否启用语音
    :param tts_duration: 语音播放时长
    :param tts_interval: 语音触发间隔
    :param tts_engine: 语音合成引擎
    :param tts_state: 语音触发状态
    :param tts_text: 语音触发文本
    :return:
    """
    if not tts_enable or not tts_engine:
        logger.info("语音合成引擎未加载or语言合成引擎初始化失败")
        return
    classes = result.boxes.cls.cpu().numpy() if result.boxes else []
    # 检查是否有佩戴安全帽的检测结果
    has_person = 2 in classes
    has_head = 0 in classes
    has_safe_helmet = 4 in classes
    # 获取当前时间
    current_time = time.time()
    # 检查是否处于冷却时间,具体上次提示小于冷却时间则不提示
    in_cooldown = tts_state.get('last_tts_time') and (current_time -
                            tts_state.get('last_tts_time') < tts_interval)

    if not in_cooldown:
        # 如果检测到未佩戴安全帽,则进行语音提醒
        if has_person and has_head and not has_safe_helmet:
            # 如果首次检测到未佩戴安全帽,则记录开始时间
            if tts_state.get('no_helmet_start_time') is None:
                tts_state['no_helmet_start_time'] = current_time
                logger.info(f"检测到未佩戴安全帽,开始计时: {tts_state.get('no_helmet_start_time')}")
            elif current_time - tts_state.get('no_helmet_start_time') >= tts_duration:
                # 如果未佩戴安全帽的时间超过指定时长,则进行语音提醒
                logger.info(f"未佩戴安全帽的时间超过指定时长,进行语音提醒")
                try:
                    tts_engine.say(tts_text)
                    tts_engine.runAndWait()
                    logger.info(f"语音提醒结束,当前时间: {tts_state.get('last_tts_time')}")
                    tts_state['last_tts_time'] = current_time
                    tts_state['no_helmet_start_time'] = None
                except Exception as e:
                    logger.info(f"语音提醒失败: {e}")
        else:
            # 如果检测到已佩戴安全帽,则重置未佩戴安全帽的开始时间
            if tts_state.get('no_helmet_start_time') is not None:
                logger.info(f"已佩戴安全帽,重置未佩戴安全帽的开始时间")
                tts_state['no_helmet_start_time'] = None
