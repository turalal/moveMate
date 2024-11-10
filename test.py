
def process_and_save(self, metadata_df, output_dir, num_samples=None):
    """Updated process_and_save with metadata saving"""
    # ... (previous implementation) ...
    
    try:
        # Save metadata log if exists
        if hasattr(self, 'metadata_log') and self.metadata_log:
            metadata_df = pd.DataFrame(self.metadata_log)
            metadata_df.to_csv(
                output_dir / 'dicom_metadata.csv',
                index=False
            )
            print(f"\nDICOM metadata saved for {len(self.metadata_log)} files")
            
        # Save error log with additional details
        if self.error_log:
            error_df = pd.DataFrame(self.error_log)
            error_summary = error_df['type'].value_counts()
            error_df.to_csv(output_dir / 'processing_errors.csv', index=False)
            print("\nError Summary:")
            print(error_summary)
            
    except Exception as e:
        print(f"Error saving logs: {str(e)}")
    
    return processed_count, failed_count