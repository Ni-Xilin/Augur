# Augur.py
# -*- coding: utf-8 -*-

# 导入必要的库
import os
import struct
from twisted.internet import defer, threads # <--- 导入Twisted线程支持
from obfsproxy.transports.base import BaseTransport
from obfsproxy.common import transport_config
import obfsproxy.common.aes as aes
import obfsproxy.common.dh as dh
import obfsproxy.common.rand as rand
import obfsproxy.common.log as logging

# === 核心：从我们独立的模块中导入生成器 ===
from obfsproxy.transports.model.generator import TSTGenerator as PerturbationGenerator
import numpy as np
import collections 

# === Augur 可插拔传输实现 ===

class AugurTransport(BaseTransport):
    """
    Augur 传输协议的实现。
    """
    def __init__(self):
        super(AugurTransport, self).__init__()

    @classmethod
    def setup(cls, config):
        """
        PT的设置钩子，在这里可以进行一些一次性的全局初始化。
        """
        pass

    class client(BaseTransport.client):
        """
        Augur 客户端实现。
        """
        def __init__(self):
            super(AugurTransport.client, self).__init__()
            self.config = transport_config.ClientTransportConfig()
            self.handshake_done = False # 握手完成标志
            self.key_negotiator = None # 密钥协商器
            self.encryptor = None # 加密器
            self.decryptor = None # 解密器
            self.data_buffer = b'' # 用于在握手完成前缓存来自Tor的数据

            # --- 核心扰动生成组件 ---
            # 1. 定义模型参数和权重路径
            # !!! 重要: 请务必将此路径修改为您自己训练好的模型权重文件路径 !!!
            MODEL_CHECKPOINT_PATH = "path/to/your/generator_checkpoint.pth"
            self.SEQ_LEN = 200 # 历史序列长度，必须与您训练时使用的seq_len一致
            self.PRED_LEN = 100 # 预测序列长度，必须与您训练时使用的pred_len一致
            FEATURE_DIM = 2 # 特征维度 (IPD, size)
            
            # 2. 实例化扰动生成器
            logging.info("Augur 客户端: 正在初始化扰动生成器...")
            self.generator = PerturbationGenerator(seq_len=self.args.seq_len, 
                                patch_len=self.args.patch_len,
                                pred_len=self.args.pred_len,
                                feat_dim=self.args.enc_in, 
                                depth=self.args.depth, 
                                scale_factor=self.args.scale_factor, 
                                n_layers=self.args.n_layers, 
                                d_model=self.args.d_model, 
                                n_heads=self.args.n_heads,
                                individual=self.args.individual, 
                                d_k=None, d_v=None, 
                                d_ff=self.args.d_ff, 
                                norm='BatchNorm', 
                                attn_dropout=self.args.att_dropout, 
                                head_dropout=self.args.head_dropout, 
                                act=self.args.activation,pe='zeros', 
                                learn_pe=True,pre_norm=False, 
                                res_attention=False, 
                                store_attn=False)
            logging.info("Augur 客户端: 扰动生成器初始化完成。")
            
            # 3. 历史流量缓冲区
            # 使用一个固定大小的numpy数组作为环形缓冲区，以提高效率
            self.history_buffer = np.zeros((self.SEQ_LEN, FEATURE_DIM))
            self.last_packet_time = 0.0 # 用于计算IPD
            
            # --- 异步生成机制核心组件 ---
            # 4. 使用双端队列存储扰动，线程安全且高效
            self.perturbation_queue = collections.deque()
            # 5. 生成新扰动的触发水位（当队列剩余少于一半时）
            self.generation_trigger_level = self.PRED_LEN // 2
            # 6. 状态标志，防止同时启动多个生成线程
            self.is_generating = False
            
            # 7. 获取Twisted的事件循环reactor，用于实现异步任务
            from twisted.internet import reactor
            self.reactor = reactor

        def _background_generate(self):
            """
            【工作线程】这个函数会在一个独立的线程中执行。
            它的任务是调用昂贵的模型推理，生成一批新的扰动。
            """
            logging.info("Augur 客户端: [工作线程] 开始生成新的一批扰动...")
            # 复制历史缓冲区，避免在推理时被主线程修改
            history_snapshot = self.history_buffer.copy()
            new_perturbations = self.generator.generate(history_snapshot)
            logging.info(f"Augur 客户端: [工作线程] 成功生成 {len(new_perturbations)} 个新扰动。")
            return new_perturbations

        def _on_generation_complete(self, new_perturbations):
            """
            【主线程】当工作线程完成任务后，这个回调函数会在主线程中被调用。
            """
            logging.info("Augur 客户端: [主线程] 收到新的扰动，正在添加到队列...")
            # 将numpy数组转换为字典列表，并添加到队列末尾
            for p in new_perturbations:
                # [IPD扰动, Size扰动]，并确保为正值
                self.perturbation_queue.append({'delay_ms': abs(p[0]), 'padding_len': int(abs(p[1]))})
            
            # 重置状态标志，允许下一次生成
            self.is_generating = False

        def _on_generation_failed(self, failure):
            """
            【主线程】如果工作线程发生错误，这个回调会被调用。
            """
            logging.warning(f"Augur 客户端: [主线程] 扰动生成失败: {failure.getErrorMessage()}")
            # 同样需要重置状态标志
            self.is_generating = False

        def _trigger_perturbation_generation(self):
            """
            【主线程】检查是否需要并启动一个新的后台扰动生成任务。
            """
            # 如果队列中的扰动数量低于触发水位，并且当前没有正在进行的生成任务
            if len(self.perturbation_queue) <= self.generation_trigger_level and not self.is_generating:
                logging.info(f"Augur 客户端: [主线程] 扰动队列低于水位 ({len(self.perturbation_queue)}/{self.generation_trigger_level})，触发新的生成任务。")
                # 设置状态标志，防止重复触发
                self.is_generating = True
                # 将 _background_generate 函数交给Twisted的线程池执行，返回一个Deferred对象
                d = threads.deferToThread(self._background_generate)
                # 绑定成功和失败的回调函数
                d.addCallback(self._on_generation_complete)
                d.addErrback(self._on_generation_failed)

        def receivedDownstream(self, data):
            """
            从网络(服务器)接收数据，解密后发往Tor。
            """
            if not self.handshake_done:
                self.handle_server_handshake(data)
                return
            decrypted = self.decryptor.decrypt(data)
            self.circuit.upstream.write(decrypted)

        def receivedUpstream(self, data):
            """
            从Tor接收数据，扰动、封装、加密后发往网络。这是客户端的核心逻辑。
            """
            if not self.handshake_done:
                self.data_buffer += data
                return
            
            real_data = data.read()
            if not real_data:
                return # 如果没有真实数据，则不进行任何操作
            
            current_time = self.reactor.seconds()
            ipd = (current_time - self.last_packet_time) * 1000 if self.last_packet_time > 0 else 0
            self.last_packet_time = current_time
            
            new_feature = np.array([[ipd, len(real_data)]])
            self.history_buffer = np.roll(self.history_buffer, -1, axis=0)
            self.history_buffer[-1, :] = new_feature
            
            # 1. 从队列左侧取出一个扰动来应用
            if self.perturbation_queue:
                perturbation_cmd = self.perturbation_queue.popleft()
            else:
                logging.warning("Augur 客户端: [主线程] 扰动队列为空，使用零扰动。")
                perturbation_cmd = {'delay_ms': 0, 'padding_len': 0}

            # 2. 在消费后，立刻检查是否需要触发新的生成任务
            self._trigger_perturbation_generation()
            
            # --- 后续封装与发送逻辑 ---
            padding_len = perturbation_cmd['padding_len']
            delay_ms = perturbation_cmd['delay_ms']

            real_data_len = len(real_data)
            len_header = struct.pack("!H", real_data_len)
            padding = os.urandom(padding_len)
            plaintext_payload = len_header + real_data + padding
            
            encrypted_payload = self.encryptor.encrypt(plaintext_payload)

            delay_sec = delay_ms / 1000.0
            def send_data():
                encrypted_len_header = struct.pack("!H", len(encrypted_payload))
                self.circuit.downstream.write(encrypted_len_header + encrypted_payload)
            self.reactor.callLater(delay_sec, send_data)

        def client_handshake(self):
            """
            客户端发起握手。
            """
            self.key_negotiator = dh.KeyNegotiator()
            public_key = self.key_negotiator.get_public_key()
            self.circuit.downstream.write(b'\xDE\xC0\xAD\xDE' + public_key)
            logging.info("Augur 客户端: 发起握手...")

        def handle_server_handshake(self, data):
            """
            处理服务器的握手响应。
            """
            if len(data) < 132 or data[:4] != b'\xC0\xDE\xDA\xFE':
                logging.warning("Augur 客户端: 握手失败，收到无效的服务器响应。")
                self.circuit.close()
                return

            server_public_key = data[4:132]
            shared_secret = self.key_negotiator.get_shared_secret(server_public_key)
            send_key = aes.generate_key_iv_from_secret(shared_secret, b"Augur_send")[0]
            recv_key = aes.generate_key_iv_from_secret(shared_secret, b"Augur_recv")[0]
            self.encryptor = aes.AES_CTR(send_key, os.urandom(16))
            self.decryptor = aes.AES_CTR(recv_key, os.urandom(16))
            self.handshake_done = True
            logging.info("Augur 客户端: 握手成功，连接已建立。")
            
            # 握手成功后，立即触发第一次扰动生成，以“填满”初始队列
            logging.info("Augur 客户端: [主线程] 启动初始扰动生成...")
            self._trigger_perturbation_generation()

            # 发送之前缓存的数据
            if self.data_buffer:
                self.receivedUpstream(self.data_buffer)
                self.data_buffer = b''

        def transport_open(self):
            """
            当传输被打开时，发起握手。
            """
            super(AugurTransport.client, self).transport_open()
            self.client_handshake()


    class server(BaseTransport.server):
        """
        Augur 服务器端实现。
        """
        def __init__(self):
            super(AugurTransport.server, self).__init__()
            self.handshake_done = False
            self.key_negotiator = None
            self.encryptor = None
            self.decryptor = None
            self.recv_buffer = b''
            
        def receivedDownstream(self, data):
            """
            从Tor接收数据，加密后发往客户端。
            """
            if not self.handshake_done:
                return
            encrypted = self.encryptor.encrypt(data)
            self.circuit.upstream.write(encrypted)

        def receivedUpstream(self, data):
            """
            从网络(客户端)接收数据，解扰、解析后发往Tor。这是服务器端的核心逻辑。
            """
            if not self.handshake_done:
                self.handle_client_handshake(data)
                return
            
            self.recv_buffer += data
            
            while True:
                if len(self.recv_buffer) < 2:
                    break 
                encrypted_payload_len = struct.unpack("!H", self.recv_buffer[:2])[0]
                if len(self.recv_buffer) < 2 + encrypted_payload_len:
                    break 
                encrypted_payload = self.recv_buffer[2 : 2 + encrypted_payload_len]
                decrypted_payload = self.decryptor.decrypt(encrypted_payload)
                if len(decrypted_payload) < 2:
                    logging.warning("Augur 服务器: 解密后的载荷过短，无法解析。")
                    self.recv_buffer = self.recv_buffer[2 + encrypted_payload_len:]
                    continue
                real_data_len = struct.unpack("!H", decrypted_payload[:2])[0]
                real_data = decrypted_payload[2 : 2 + real_data_len]
                self.circuit.downstream.write(real_data)
                self.recv_buffer = self.recv_buffer[2 + encrypted_payload_len:]

        def handle_client_handshake(self, data):
            """
            处理客户端的握手请求。
            """
            if len(data) < 132 or data[:4] != b'\xDE\xC0\xAD\xDE':
                logging.warning("Augur 服务器: 握手失败，收到无效的客户端请求。")
                self.circuit.close()
                return
            client_public_key = data[4:132]
            self.key_negotiator = dh.KeyNegotiator()
            shared_secret = self.key_negotiator.get_shared_secret(client_public_key)
            server_public_key = self.key_negotiator.get_public_key()
            send_key = aes.generate_key_iv_from_secret(shared_secret, b"Augur_recv")[0]
            recv_key = aes.generate_key_iv_from_secret(shared_secret, b"Augur_send")[0]
            self.encryptor = aes.AES_CTR(send_key, os.urandom(16))
            self.decryptor = aes.AES_CTR(recv_key, os.urandom(16))
            self.circuit.upstream.write(b'\xC0\xDE\xDA\xFE' + server_public_key)
            self.handshake_done = True
            logging.info("Augur 服务器: 握手成功，连接已建立。")