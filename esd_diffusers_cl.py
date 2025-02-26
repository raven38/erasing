import os
import pandas as pd
import torch
import argparse
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from PIL import Image
from matplotlib import pyplot as plt
import textwrap
import argparse
import torch
import copy
import os
import re

from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from utils.utils import *


class ContinuousESDTrainer:
    """
    複数のコンセプトを継続的に忘却させる学習フレームワーク
    """
    def __init__(
        self,
        concept_csv_path,
        evaluation_prompts_csv_path,
        output_dir,
        model_path="CompVis/stable-diffusion-v1-4",
        train_method="xattn",
        iterations=200,
        negative_guidance=1.0,
        lr=2e-5,
        device="cuda:0",
        group_lasso_weight=0.0,
        rank=4,
        evaluation_prompts=None,
        eval_during_training=False,
        save_intermediate=True,
        seed=42
    ):
        self.concept_csv_path = concept_csv_path
        self.evaluation_prompts_csv_path = evaluation_prompts_csv_path
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.train_method = train_method
        self.iterations = iterations
        self.negative_guidance = negative_guidance
        self.lr = lr
        self.device = device
        self.group_lasso_weight = group_lasso_weight
        self.rank = rank
        self.evaluation_prompts = evaluation_prompts
        self.eval_during_training = eval_during_training
        self.save_intermediate = save_intermediate
        self.seed = seed
        
        # 出力ディレクトリを作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験設定をJSON形式で保存
        self.save_experiment_config()
        
        # ロギングの設定
        self.setup_logging()
        
        # コンセプトリストを読み込む
        self.concepts = self.load_concepts_from_csv()

        self_eval_prompts = self.load_prompts_from_csv()
        
        # シードの設定
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # 実行時間を記録
        self.start_time = datetime.now()
        
    def setup_logging(self):
        """ロギングの設定"""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_experiment_config(self):
        """実験設定をJSONとして保存"""
        config = {
            "concept_csv_path": str(self.concept_csv_path),
            "model_path": self.model_path,
            "train_method": self.train_method,
            "iterations": self.iterations,
            "negative_guidance": self.negative_guidance,
            "lr": self.lr,
            "device": self.device,
            "group_lasso_weight": self.group_lasso_weight,
            "rank": self.rank,
            "eval_during_training": self.eval_during_training,
            "save_intermediate": self.save_intermediate,
            "seed": self.seed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_concepts_from_csv(self):
        """CSVファイルからコンセプトリストを読み込む"""
        df = pd.read_csv(self.concept_csv_path)
        self.logger.info(f"Loaded {len(df)} concepts from {self.concept_csv_path}")
        return df
    
    def load_prompts_from_csv(self):
        """CSVファイルから評価用プロンプトを読み込む"""
        df = pd.read_csv(self.evaluation_prompts_csv_path)
        self.logger.info(f"Loaded {len(df)} evaluation prompts from {self.evaluation_prompts_csv_path}")
        return df
    
    def split_eval_prompts(self, training_stage):
        flag = 0
        if 'erased' in self.eval_prompts['type']:
            flag = 'erased'
        prompts_removed = self.eval_prompts[self.eval_prompts['type'] == flag & self.eval_prompts['stage'] <= training_stage]
        prompts_kept = self.eval_prompts[self.eval_prompts['type'] != flag | self.eval_prompts['stage'] > training_stage]
        return prompts_removed, prompts_kept
    
    def initialize_model(self):
        """モデルの初期化"""
        # StableDiffuserモデルを初期化
        self.logger.info(f"Initializing model from {self.model_path}")
        diffuser = StableDiffuser(scheduler='DDIM', model_path=self.model_path).to(self.device)
        diffuser.train()
        return diffuser
        
    def train_single_concept(self, diffuser, concept_row, checkpoint_path=None):
        """
        単一のコンセプトに対する忘却学習を実行
        
        Args:
            diffuser: StableDiffuserモデル
            concept_row: コンセプト情報の行（DataFrameの1行）
            checkpoint_path: 以前のチェックポイントがあれば指定
            
        Returns:
            finetuner: 訓練済みのFineTunedModelLoRAモデル
        """
        concept_id = concept_row['id']
        concept = concept_row['concept']
        category = concept_row['category']
        
        self.logger.info(f"Training concept {concept_id}: {concept} (Category: {category})")
        
        # LoRAのfinetunerを初期化
        finetuner = FineTunedModelLoRA(diffuser, train_method=self.train_method, rank=self.rank)
        
        # 以前のチェックポイントがあれば読み込む
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.logger.info(f"Loading previous checkpoint from {checkpoint_path}")
            finetuner.load_state_dict(torch.load(checkpoint_path))
        
        # オプティマイザを設定
        optimizer = torch.optim.Adam(finetuner.parameters(), lr=self.lr)
        criteria = torch.nn.MSELoss()
        
        # 訓練ループ
        pbar = tqdm(range(self.iterations))
        
        # コンセプトを消去する設定
        # この例では、コンセプトを完全に消去する（コンセプト => ""）
        erase_concept = [concept]
        erase_from = [""]  # 空文字列に変換（完全消去）
        
        # コンセプトペアをリスト形式に変換
        erase_concept_pairs = [[e, f] for e, f in zip(erase_concept, erase_from)]
        
        # 訓練ループ
        for i in pbar:
            with torch.no_grad():
                # コンセプトをサンプリング（この場合は1つしかない）
                index = 0
                erase_concept_sampled = erase_concept_pairs[index]
                
                # テキスト埋め込みを取得
                neutral_text_embeddings = diffuser.get_text_embeddings([''], n_imgs=1)
                positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]], n_imgs=1)
                target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]], n_imgs=1)
                
                # タイムステップのセットアップ
                nsteps = 50
                diffuser.set_scheduler_timesteps(nsteps)
                
                optimizer.zero_grad()
                
                # ランダムなタイムステップをサンプリング
                iteration = torch.randint(1, nsteps - 1, (1,)).item()
                
                # 初期latentsを取得
                latents = diffuser.get_initial_latents(1, 512, 1)
                
                # finetunerを使って拡散過程を実行
                with finetuner:
                    latents_steps, _ = diffuser.diffusion(
                        latents,
                        positive_text_embeddings,
                        start_iteration=0,
                        end_iteration=iteration,
                        guidance_scale=3,
                        show_progress=False
                    )
                
                # 1000ステップのスケジューラにセット
                diffuser.set_scheduler_timesteps(1000)
                
                # タイムステップを変換
                iteration = int(iteration / nsteps * 1000)
                
                # 各条件でのノイズ予測を取得
                positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
                neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
                target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
                
                # 同じコンセプトの場合、ニュートラルlatentsをターゲットとする
                if erase_concept_sampled[0] == erase_concept_sampled[1]:
                    target_latents = neutral_latents.clone().detach()
            
            # finetunerでノイズを予測
            with finetuner:
                negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            
            # 参照latentsの勾配計算を無効化
            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            
            # ESD損失を計算
            esd_loss = criteria(negative_latents, target_latents - (self.negative_guidance * (positive_latents - neutral_latents)))
            
            # Group Lasso損失の計算（有効な場合）
            if self.group_lasso_weight > 0.0:
                with finetuner:
                    group_lasso_loss = calculate_group_lasso_loss(diffuser.unet) * self.group_lasso_weight
            else:
                group_lasso_loss = 0.0
            
            # 全体の損失関数
            loss = esd_loss + group_lasso_loss
            
            # 学習ステップの実行
            loss.backward()
            optimizer.step()
            
            # プログレスバーの更新
            pbar.set_postfix({
                "esd_loss": esd_loss.item(),
                "lasso_loss": group_lasso_loss.item() if isinstance(group_lasso_loss, torch.Tensor) else group_lasso_loss
            })
            
            # 学習中評価を実行（設定されていれば）
            if self.eval_during_training and (i + 1) % (self.iterations // 4) == 0:
                self.evaluate_model(diffuser, finetuner, concept_row, iteration=i)
        
        return finetuner
    
    def inference_prompts(self, diffuser, finetuner, prompts, save_path):
        save_path.mkdir(parents=True, exist_ok=True)
        for i, prompt_row in prompts.iterrows():
            prompt = prompt_row['prompt']
            evaluation_seed = prompt_row['evaluation_seed']
            with torch.no_grad(), finetuner:
                torch.manual_seed(evaluation_seed)
                images = diffuser(prompt, n_steps=50, guidance_scale=7.5)
                for j, img in enumerate(images[0]):
                    img_path = save_path / f"{prompt}_{evaluation_seed}.png"
                    img.save(img_path)
                    self.logger.info(f"Saved evaluation image to {img_path}")

    def evaluate_model_stage(self, diffuser, finetuner, concept_row, iteration=None):
        concept = concept_row['concept']
        concept_id = concept_row['id']

        eval_base_dir = self.output_dir / f"evaluation/concept_{concept_id}_{concept}/"
        eval_removed_dir = eval_base_dir / "removed"
        eval_kept_dir = eval_base_dir / "kept"

        eval_removed_dir.mkdir(parents=True, exist_ok=True)
        eval_kept_dir.mkdir(parents=True, exist_ok=True)

        self.inference_prompts(diffuser, finetuner, prompts_removed, eval_removed_dir)
        self.inference_prompts(diffuser, finetuner, prompts_kept, eval_kept_dir)

    
    def evaluate_model(self, diffuser, finetuner, concept_row, iteration=None):
        """
        モデルの評価を実行
        
        Args:
            diffuser: StableDiffuserモデル
            finetuner: 訓練済みのFineTunedModelLoRAモデル
            concept_row: 評価中のコンセプト情報
            iteration: 現在の学習ステップ（Noneの場合は最終評価）
        """
        concept = concept_row['concept']
        concept_id = concept_row['id']
        
        # 評価用プロンプトの設定
        # ここでは簡単な評価用プロンプトを設定していますが、実際には外部ファイルから読み込むなどが良いでしょう
        eval_prompts = self.evaluation_prompts
        if eval_prompts is None:
            eval_prompts = [
                f"a photo of {concept}",
                f"{concept} in nature",
                f"{concept} portrait"
            ]
        
        self.logger.info(f"Evaluating model on concept: {concept}")
        
        # 評価結果を保存するディレクトリ
        if iteration is not None:
            eval_dir = self.output_dir / f"evaluation/concept_{concept_row['id']}_{concept}/iteration_{iteration}"
        else:
            eval_dir = self.output_dir / f"evaluation/concept_{concept_row['id']}_{concept}/final"
        
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 評価用の画像生成
        for i, prompt in enumerate(eval_prompts):
            # 画像を生成
            with torch.no_grad(), finetuner:
                images = diffuser(prompt, n_steps=50, guidance_scale=7.5)
                
                # 画像を保存
                for j, img in enumerate(images[0]):
                    img_path = eval_dir / f"prompt_{i}_image_{j}.png"
                    img.save(img_path)
                    self.logger.info(f"Saved evaluation image to {img_path}")
        
        # メトリクスの計算と保存（ここでは簡易的なもののみ）
        eval_metrics = {
            "concept": concept,
            "iteration": iteration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # メトリクスをJSONとして保存
        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
    
    def run_continuous_training(self):
        """
        継続学習の実行
        複数のコンセプトに対して順番に学習を行う
        """
        self.logger.info("Starting continuous training")
        
        # モデルの初期化
        diffuser = self.initialize_model()
        
        # チェックポイント保存用の変数
        prev_checkpoint = None
        
        # 各コンセプトに対して学習を実行
        for i, concept_row in self.concepts.iterrows():
            concept_id = concept_row['id']
            concept = concept_row['concept']
            
            # コンセプトごとのチェックポイントパス
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"checkpoint_concept_{concept_id}_{concept}.pt"
            
            # 現在のコンセプトに対して学習
            finetuner = self.train_single_concept(diffuser, concept_row, checkpoint_path=prev_checkpoint)
            
            # 学習後の評価
            # TODO split eval prompts into deleted concept and keep conept
            self.evaluate_model_stage(diffuser, finetuner, concept_row)
            
            # チェックポイントを保存
            torch.save(finetuner.state_dict(), checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # 次のコンセプト学習のために現在のチェックポイントを記録
            prev_checkpoint = checkpoint_path
        
        # 訓練完了
        self.logger.info("Continuous training completed")
        self.logger.info(f"Total training time: {datetime.now() - self.start_time}")
        
        return prev_checkpoint


def main():
    parser = argparse.ArgumentParser(
        prog="ContinuousESDLoRA",
        description="Continuous learning framework for erasing concepts with LoRA"
    )
    
    parser.add_argument('--concept_csv', type=str, required=True, help='Path to CSV file with concepts')
    parser.add_argument('--prompt_csv', type=str, default=None, help='Path to CSV file with evaluation prompts')
    parser.add_argument('--output_dir', type=str, default='output/continuous_esd', help='Output directory')
    parser.add_argument('--model_path', type=str, default="CompVis/stable-diffusion-v1-4", help='Base model path')
    parser.add_argument('--train_method', type=str, default="xattn", help='Training method (xattn, noxattn, full, etc.)')
    parser.add_argument('--iterations', type=int, default=200, help='Training iterations per concept')
    parser.add_argument('--negative_guidance', type=float, default=1.0, help='Negative guidance value')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to train on')
    parser.add_argument('--group_lasso_weight', type=float, default=0.0, help='Group Lasso regularization weight')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--eval_during_training', action='store_true', help='Whether to evaluate during training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # トレーナーの初期化と実行
    trainer = ContinuousESDTrainer(
        concept_csv_path=args.concept_csv,
        evaluation_prompts_csv_path=args.prompt_csv,
        output_dir=args.output_dir,
        model_path=args.model_path,
        train_method=args.train_method,
        iterations=args.iterations,
        negative_guidance=args.negative_guidance,
        lr=args.lr,
        device=args.device,
        group_lasso_weight=args.group_lasso_weight,
        rank=args.rank,
        eval_during_training=args.eval_during_training,
        seed=args.seed
    )
    
    # 継続学習の実行
    final_checkpoint = trainer.run_continuous_training()
    print(f"Training completed. Final checkpoint saved to: {final_checkpoint}")


if __name__ == "__main__":
    main()