Download the entire folder as Zip, and import it into your go/fevm workspace.

Connect to serverless compute, run all cells, and follow the instructions in cells 10 and 11.

This creates a security group. We will put identities into this security group so we only need to grant access to one thing.

It will also create a service principal for you to use as the App's identity when running it locally. This lets you use the same code for local development as when its deployed.

It will create a lakebase instance, and assign relevant permissions to the group.

It will also create an 'App' for us to deploy to.