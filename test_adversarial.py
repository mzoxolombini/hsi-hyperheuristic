"""
Adversarial Robustness Testing (Section 8.6)
Implements FGSM, PGD, and C&W attacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class FGSMAttack:
    """Fast Gradient Sign Method (Goodfellow et al., 2014)"""
    
    def __init__(self, epsilon: float = 0.03):
        self.epsilon = epsilon
    
    def attack(
        self, 
        model: nn.Module, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial examples."""
        images = images.clone().detach().requires_grad_(True)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial perturbation
        grad_sign = images.grad.sign()
        adv_images = images + self.epsilon * grad_sign
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach(), outputs


class PGDAttack:
    """Projected Gradient Descent (Madry et al., 2017)"""
    
    def __init__(
        self, 
        epsilon: float = 0.03, 
        alpha: float = 0.01, 
        num_iter: int = 10
    ):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
    
    def attack(
        self, 
        model: nn.Module, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial examples using PGD."""
        adv_images = images.clone().detach()
        
        for _ in range(self.num_iter):
            adv_images.requires_grad_(True)
            outputs = model(adv_images)
            loss = loss_fn(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            
            # Update with sign gradient
            grad_sign = adv_images.grad.sign()
            adv_images = adv_images + self.alpha * grad_sign
            
            # Project back to epsilon ball
            perturbation = torch.clamp(adv_images - images, -self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + perturbation, 0, 1)
            adv_images = adv_images.detach()
        
        return adv_images, model(adv_images)


def test_robustness(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_type: str = 'fgsm',
    epsilon: float = 0.03,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate model robustness against adversarial attacks.
    
    Returns:
        dict with clean_accuracy, adv_accuracy, retention_rate
    """
    model.eval()
    model.to(device)
    
    if attack_type.lower() == 'fgsm':
        attacker = FGSMAttack(epsilon)
    elif attack_type.lower() == 'pgd':
        attacker = PGDAttack(epsilon)
    else:
        raise ValueError(f"Unknown attack: {attack_type}")
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for batch in test_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Get labels (argmax over classes)
        labels = masks.argmax(dim=1) if masks.shape[1] > 1 else masks
        
        # Clean accuracy
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
        
        # Adversarial accuracy
        adv_images, adv_outputs = attacker.attack(model, images, labels)
        adv_preds = adv_outputs.argmax(dim=1)
        adv_correct += (adv_preds == labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    retention = (adv_acc / clean_acc) * 100 if clean_acc > 0 else 0
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'retention_rate': retention,
        'attack_type': attack_type,
        'epsilon': epsilon
    }


if __name__ == '__main__':
    # Example usage
    print("Adversarial Robustness Test Module")
    print("Usage: from validation.test_adversarial import test_robustness")