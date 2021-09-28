import os
import wandb
from dotenv import load_dotenv


class Wandb:
    def __init__(self, config):
        """
        wandb config의 auth key값을 사용하여 로그인
        """
        self.config = config
        self.wandb_config = config["wandb"]
        self.unique_tag = self.wandb_config["unique_tag"]
        self.entity = self.wandb_config["entity"]
        self.project = self.wandb_config["project"]

        dotenv_path = self.wandb_config["env_path"]
        load_dotenv(dotenv_path=dotenv_path)
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

    def init_wandb(self, phase="train", **kwargs):
        """
        :param phase: 'train' or 'valid'
        :param args: arguments변수
        :param **kwargs: wandb의 태그와 name에 추가하고싶은 내용을 넣어줌 ex_ fold=1
        """
        tags = [f"unique_tag: {self.unique_tag}", f"{phase}"]
        name = f"{phase}, {self.unique_tag}"

        if phase != "team_eval":
            tags.extend(
                [
                    f"name: {self.config.name}",
                    f"lr: {self.config.optimizer.args.lr}",
                    f"optimizer: {self.config.optimizer.type}",
                    f"usertrans: {self.config.data_loader.type}",
                ]
            )
            name = f"{self.config.arch.type}, {phase}, {self.unique_tag}"
        if kwargs:
            for k, v in kwargs.items():
                tags.append(f"{k}: {v}")
                name += f" {v}{k} "
        wandb.init(tags=tags, entity=self.entity, project=self.project, reinit=True)
        wandb.run.name = name
        wandb.config.update(self.config)
        wandb.config.update({"PHASE": phase})

    def log_wandb(phase="train", acc=0, loss=0, single_table=False):
        """
        wandb에 차트 그래프를 그리기 위해 로그를 찍는 함수
        :param phase: 'train' or 'valid'
        :paramacc: accuracy
        :param loss: loss
        :param sing_table: True로 체크하면 train과 validation이 한개의 차트에 표시됨
        """

        if single_table == True:
            log = {acc: acc, "loss": loss}
        else:
            log = {f"{phase}_acc": acc, f"{phase}_loss": loss}

        # log = {f"{phase}_acc": acc, f"{phase}_f1_score": f1_score}
        # if phase != "team_eval":
        #     log[f"{phase}_loss"] = loss

        # if single_table == True:
        #     log = {acc: acc, "f1_score": f1_score, "loss": loss}

        # if phase == "best":
        #     log = {f"{phase}_f1_score": f1_score}

        wandb.log(log)

    # def show_images_wandb(images, y_labels, preds):
    #     """
    #     wandb에 media로 이미지를 출력함

    #     :param images: image array를 받음 [batch,channel,width,height]
    #     :param y_labels: 실제 라벨 데이터
    #     :param preds: 예측한 데이터
    #     """
    #     for i in range(len(y_labels)):
    #         im = images[i, :, :, :]
    #         im = im.permute(1, 2, 0).cuda().cpu().detach().numpy()
    #         wandb.log(
    #             {
    #                 "image_preds": [
    #                     wandb.Image(
    #                         im, caption=f"real: {y_labels[i]}, predict: {preds[i]}"
    #                     )
    #                 ]
    #             }
    #         )
    # my_table = wandb.Table()
    # my_table.add_column("image", wandb.Image(im))
    # my_table.add_column("label", y_labels)
    # my_table.add_column("class_prediction", preds)
    # wandb.log({"image_preds_table": my_table})
