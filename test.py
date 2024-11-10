def train_model():
    """Main training loop with enhanced error handling and monitoring"""
    # Set seeds for reproducibility
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize training metrics and monitor
    fold_scores = []
    best_scores = []
    monitor = TrainingMonitor(CFG)
    
    try:
        # Read and prepare data
        train_df = pd.read_csv(CFG.base_path / 'train.csv')
        if CFG.debug:
            train_df = train_df.head(100)
            print("Debug mode: Using only 100 training samples")
        
        # Print initial class distribution
        print("\nOverall class distribution:")
        print(train_df['cancer'].value_counts(normalize=True))
        
        # Create stratified folds
        skf = StratifiedGroupKFold(
            n_splits=CFG.num_folds, 
            shuffle=True, 
            random_state=CFG.seed
        )
        
        # Create folds while maintaining patient groups
        train_df['fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df, train_df['cancer'], groups=train_df['patient_id'])
        ):
            train_df.loc[val_idx, 'fold'] = fold
        
        # Save fold assignments for reproducibility
        train_df.to_csv(CFG.processed_dir / 'fold_assignments.csv', index=False)
        
        # Training loop for each fold
        for fold in range(CFG.num_folds):
            print(f'\n{"="*20} Fold {fold + 1}/{CFG.num_folds} {"="*20}')
            
            # Clear memory before each fold
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Prepare fold data
                train_fold = train_df[train_df.fold != fold].reset_index(drop=True)
                valid_fold = train_df[train_df.fold == fold].reset_index(drop=True)
                
                # Print fold-specific class distributions
                print("\nTrain fold class distribution:")
                print(train_fold['cancer'].value_counts(normalize=True))
                print("\nValid fold class distribution:")
                print(valid_fold['cancer'].value_counts(normalize=True))
                
                # Create datasets
                train_dataset = RSNADataset(
                    train_fold, 
                    transform=A.Compose(CFG.train_aug_list)
                )
                valid_dataset = RSNADataset(
                    valid_fold, 
                    transform=A.Compose(CFG.valid_aug_list)
                )
                
                # Create dataloaders with safe settings
                train_loader = get_dataloader(
                    train_dataset, 
                    CFG.train_batch_size, 
                    shuffle=True
                )
                valid_loader = get_dataloader(
                    valid_dataset, 
                    CFG.valid_batch_size, 
                    shuffle=False, 
                    is_train=False
                )
                
                # Initialize model and move to device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = RSNAModel().to(device)
                
                # Print model summary
                monitor.print_model_summary(model)
                
                # Calculate class weights for imbalanced dataset
                pos_weight = (
                    len(train_fold[train_fold['cancer'] == 0]) / 
                    max(1, len(train_fold[train_fold['cancer'] == 1]))
                )
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight]).to(device)
                )
                
                # Initialize optimizer
                optimizer = getattr(torch.optim, CFG.optimizer)(
                    model.parameters(),
                    lr=CFG.learning_rate,
                    weight_decay=CFG.weight_decay
                )
                
                # Initialize scheduler
                scheduler = getattr(torch.optim.lr_scheduler, CFG.scheduler)(
                    optimizer,
                    T_max=CFG.T_max,
                    eta_min=CFG.min_lr
                )
                
                # Training loop
                best_score = float('-inf')
                best_loss = float('inf')
                patience_counter = 0
                fold_history = {
                    'train_loss': [],
                    'valid_loss': [],
                    'valid_score': []
                }
                
                for epoch in range(CFG.epochs):
                    print(f'\nEpoch {epoch + 1}/{CFG.epochs}')
                    
                    # Monitor GPU usage at start of epoch
                    gpu_stats = monitor.monitor_gpu_usage()
                    if gpu_stats:
                        print("\nGPU Memory Usage:")
                        for stat in gpu_stats:
                            print(f"Device {stat['device']}: "
                                  f"{stat['allocated_mb']:.0f}MB allocated, "
                                  f"{stat['reserved_mb']:.0f}MB reserved")
                    
                    try:
                        # Training phase
                        train_loss = train_one_epoch(
                            model, train_loader, criterion,
                            optimizer, scheduler, device
                        )
                        
                        # Validation phase
                        valid_loss, valid_score = valid_one_epoch(
                            model, valid_loader, criterion, device
                        )
                        
                        # Update history
                        fold_history['train_loss'].append(train_loss)
                        fold_history['valid_loss'].append(valid_loss)
                        fold_history['valid_score'].append(valid_score)
                        
                        # Save batch metrics
                        monitor.save_batch_metrics(
                            epoch, 
                            {
                                'train_loss': train_loss,
                                'valid_loss': valid_loss,
                                'valid_score': valid_score
                            },
                            fold,
                            epoch
                        )
                        
                        # Print metrics
                        print(
                            f'Train Loss: {train_loss:.4f} '
                            f'Valid Loss: {valid_loss:.4f} '
                            f'Valid Score: {valid_score:.4f}'
                        )
                        
                        # Save best model and visualize
                        if valid_score > best_score:
                            best_score = valid_score
                            best_loss = valid_loss
                            
                            # Save model
                            torch.save(
                                {
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'best_score': best_score,
                                    'best_loss': best_loss,
                                    'fold_history': fold_history
                                },
                                CFG.model_dir / f'fold{fold}_best.pth'
                            )
                            print(f'Best model saved! Score: {best_score:.4f}')
                            
                            # Visualization and logging
                            fig = monitor.visualize_training_progress(fold_history)
                            fig.savefig(monitor.monitor_dir / f'fold{fold}_training_progress.png')
                            plt.close(fig)
                            
                            monitor.log_fold_metrics(fold, fold_history, best_score, best_loss)
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            
                        # Early stopping check
                        if patience_counter >= CFG.patience:
                            print(f'Early stopping triggered after {epoch + 1} epochs')
                            break
                            
                    except Exception as e:
                        print(f"Error in epoch {epoch + 1}: {str(e)}")
                        print(traceback.format_exc())
                        break
                
                # Store fold results and save predictions
                fold_scores.append(best_score)
                best_scores.append({
                    'fold': fold,
                    'score': best_score,
                    'loss': best_loss
                })
                
                # Save validation predictions
                valid_preds = model(valid_dataset[::]['images'].to(device)).sigmoid().cpu().numpy()
                monitor.save_fold_predictions(fold, valid_preds, valid_dataset[::]['labels'].numpy(), valid_fold)
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                print(traceback.format_exc())
                continue
                
            finally:
                # Cleanup
                try:
                    del model, train_loader, valid_loader
                    del train_dataset, valid_dataset
                    del optimizer, scheduler
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error in cleanup: {str(e)}")
        
        # Print final results
        print("\nTraining completed!")
        print("\nBest scores per fold:")
        for score_dict in best_scores:
            print(
                f"Fold {score_dict['fold'] + 1}: "
                f"Score = {score_dict['score']:.4f}, "
                f"Loss = {score_dict['loss']:.4f}"
            )
        
        print(f"\nMean CV score: {np.mean(fold_scores):.4f}")
        print(f"Std CV score: {np.std(fold_scores):.4f}")
        
        # Analyze overall predictions
        for fold in range(CFG.num_folds):
            pred_path = monitor.monitor_dir / f'fold{fold}_predictions.csv'
            if pred_path.exists():
                metrics = monitor.analyze_predictions(pred_path)
                print(f"\nFold {fold + 1} Prediction Analysis:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
        
        return best_scores
        
    except Exception as e:
        print(f"Fatal error in training: {str(e)}")
        print(traceback.format_exc())
        return None