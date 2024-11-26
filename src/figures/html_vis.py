import plotly.express as px
import umap
import os

def plot_umap_pca_html(self, method: str = "UMAP", n_components: int = 2, output_file: str = "output.html") -> None:
    if self.metadata.empty:
        print("No metadata to analyze. Run extract_metadata() first.")
        return

    # Select features for dimensionality reduction
    features = self.metadata[["Brightness", "Contrast", "Aspect_Ratio", "Entropy"]].values
    image_paths = self.metadata["Image"].values
    image_full_paths = [os.path.join(self.root_folder, img) for img in image_paths]

    # Perform dimensionality reduction
    if method.upper() == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings = reducer.fit_transform(features)
    elif method.upper() == "PCA":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        embeddings = reducer.fit_transform(features)
    else:
        print(f"Invalid method '{method}'. Choose 'UMAP' or 'PCA'.")
        return

    # Prepare data for Plotly
    data = {
        "X": embeddings[:, 0],
        "Y": embeddings[:, 1],
        "Image": image_full_paths,
        "Brightness": self.metadata["Brightness"],
        "Contrast": self.metadata["Contrast"],
        "Aspect_Ratio": self.metadata["Aspect_Ratio"],
        "Entropy": self.metadata["Entropy"]
    }

    # Create a Plotly scatter plot
    fig = px.scatter(
        data,
        x="X",
        y="Y",
        title=f"{method} Visualization of Image Features",
        hover_data=["Brightness", "Contrast", "Aspect_Ratio", "Entropy"],
        custom_data=["Image"]
    )

    # Add hoverable thumbnails
    fig.update_traces(
        marker=dict(size=8, opacity=0.8),
        hovertemplate="<b>Image:</b> %{customdata[0]}<br>" +
                      "<b>X:</b> %{x}<br>" +
                      "<b>Y:</b> %{y}<br>" +
                      "<b>Brightness:</b> %{hoverdata[0]}<br>" +
                      "<b>Contrast:</b> %{hoverdata[1]}<br>" +
                      "<b>Aspect Ratio:</b> %{hoverdata[2]}<br>" +
                      "<b>Entropy:</b> %{hoverdata[3]}<br>" +
                      "<img src='%{customdata[0]}' style='max-width:80px; max-height:80px;'>"
    )

    # Save to an HTML file
    fig.write_html(output_file)
    print(f"Interactive plot saved to {output_file}")
